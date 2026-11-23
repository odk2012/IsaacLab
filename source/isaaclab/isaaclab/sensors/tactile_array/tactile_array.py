
"""Experimental Newton/MJWarp tactile array.

CPU aggregation for small-N validation.
Forces are the final physics-substep sample, not a period average.
Contact positions are force-magnitude-weighted ankle-frame centroids.
No pressure conversion, interpolation, or policy observation is added.
"""
from dataclasses import dataclass
import re
import numpy as np
import torch
import warp as wp
import mujoco as mj
import mujoco_warp as mw
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager

@dataclass
class TactileArrayData:
    force_w: torch.Tensor
    force_b: torch.Tensor
    contact_pos_b: torch.Tensor
    taxel_pos_b: torch.Tensor
    in_contact: torch.Tensor
    valid: torch.Tensor

class TactileArray(SensorBase):
    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    def _initialize_impl(self):
        super()._initialize_impl()
        self.solver = None
        self.bindings = []
        self.world_env = {}
        centers = np.full((self._num_envs, 2, 27, 3), np.nan)
        shape = (self._num_envs, 2, 27, 3)
        self._data = TactileArrayData(
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
            torch.full(shape, float("nan"), device=self.device),
            torch.tensor(centers, dtype=torch.float32, device=self.device),
            torch.zeros(shape[:-1], dtype=torch.bool, device=self.device),
            torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
        )
        self._skip = np.ones(self._num_envs, dtype=bool)
        self.last_totals = np.zeros((self._num_envs, 2, 3))
        self.last_records = []
        self.contact_records = []


    def _bind_solver(self):
        self.solver = NewtonManager._solver
        s = self.solver
        assert getattr(s, "mjw_data", None) is not None, "Requires Newton MJWarp"
        m, d = s.mj_model, s.mjw_data
        mapping = s.mjc_geom_to_newton_shape.numpy()
        labels = getattr(s.model, "shape_label", None)
        if labels is None:
            labels = s.model.shape_key
        labels = list(labels)
        nw, ng = mapping.shape
        assert nw == self._num_envs == d.xpos.shape[0], (
            "Requires one MJWarp world per environment",
            nw, self._num_envs, d.xpos.shape
        )

        self.slot = np.full((nw, ng, 3), -1, dtype=np.int32)
        centers = np.zeros((self._num_envs, 2, 27, 3))
        seen, self.bindings = set(), []
        world_env = {}

        for w in range(nw):
            for g in range(ng):
                sh = int(mapping[w, g])
                if sh < 0:
                    continue
                name = labels[sh]
                taxel = re.search(r"/tactile_taxel_(\d{2})(?:_\d+)?$", name)
                if taxel is None:
                    continue
                env = re.search(r"/env_(\d+)/", name)
                assert env is not None, name
                e, t = int(env.group(1)), int(taxel.group(1))
                side = 0 if "/left_ankle_roll_link/" in name else 1
                assert (
                    "/left_ankle_roll_link/" if side == 0
                    else "/right_ankle_roll_link/"
                ) in name
                assert 0 <= e < self._num_envs and 0 <= t < 27
                assert (e, side, t) not in seen
                assert m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE
                assert np.isclose(m.geom_size[g, 0], 0.008)

                seen.add((e, side, t))
                assert world_env.setdefault(w, e) == e, (
                    "Mixed environments in a physics world"
                )
                self.slot[w, g] = [e, side, t]
                centers[e, side, t] = m.geom_pos[g]
                self.bindings.append(dict(
                    world=w, env=e, geom=g, shape=sh,
                    side=side, taxel=t, name=name
                ))

        assert len(seen) == self._num_envs * 54, (
            "Incomplete taxel bindings", len(seen), self._num_envs * 54
        )
        assert set(world_env.values()) == set(range(self._num_envs))
        self.world_env = world_env
        self._data.taxel_pos_b.copy_(torch.as_tensor(
            centers, dtype=torch.float32, device=self.device
        ))
        print(
            "TACTILE_SOLVER_BOUND:", len(self.bindings),
            "taxels; worlds:", self.world_env, flush=True
        )

    def reset(self, env_ids=None, env_mask=None):
        mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(env_mask=mask)
        if not hasattr(self, "_data"):
            return
        selected = mask.numpy().astype(bool)
        self._skip[selected] = True
        self._clear(selected)

    def _clear(self, selected):
        ids = torch.as_tensor(np.flatnonzero(selected), device=self.device)
        self._data.force_w[ids] = 0
        self._data.force_b[ids] = 0
        self._data.contact_pos_b[ids] = float("nan")
        self._data.in_contact[ids] = False
        self._data.valid[ids] = False
        self.contact_records = [
            r for r in self.contact_records if not selected[r["env_id"]]
        ]

    def _update_buffers_impl(self, env_mask):
        selected = env_mask.numpy().astype(bool)
        if not selected.any():
            return
        if self.solver is None:
            self._bind_solver()
        valid = selected & ~self._skip
        self._skip[selected] = False
        self._clear(selected)
        s = self.solver
        m, d = s.mj_model, s.mjw_data
        n = int(d.nacon.numpy()[0])
        assert 0 <= n < d.naconmax, "Contact capacity exhausted"
        pairs = d.contact.geom.numpy()[:n]
        worlds = d.contact.worldid.numpy()[:n]
        ids = []

        for c in range(n):
            w = int(worlds[c])
            assert 0 <= w < len(self.slot)
            if any(g >= 0 and self.slot[w, g, 0] >= 0 for g in pairs[c]):
                ids.append(c)

        f = np.zeros((self._num_envs, 2, 27, 3))
        fb, weighted = np.zeros_like(f), np.zeros_like(f)
        weights = np.zeros(f.shape[:-1])
        self.last_totals.fill(0)
        self.last_records = []
        self.contact_records = []

        if ids:
            with wp.ScopedDevice(d.contact.pos.device):
                forces = wp.zeros(len(ids), dtype=wp.spatial_vector)
                mw.contact_force(
                    s.mjw_model, d,
                    wp.array(ids, dtype=wp.int32), True, forces
                )
                raw = forces.numpy()
            assert np.isfinite(raw).all()
            pos = d.contact.pos.numpy()
            xp, xm = d.xpos.numpy(), d.xmat.numpy()

            def root_body(b):
                while m.body_parentid[b] != 0:
                    b = int(m.body_parentid[b])
                return b

            for j, c in enumerate(ids):
                w = int(worlds[c])
                for endpoint, sign in ((0, -1), (1, 1)):
                    g = int(pairs[c, endpoint])
                    other = int(pairs[c, 1-endpoint])
                    if g < 0 or self.slot[w, g, 0] < 0:
                        continue
                    e, side, t = map(int, self.slot[w, g])
                    if not valid[e]:
                        continue
                    body = int(m.geom_bodyid[g])
                    if other >= 0 and root_body(
                        int(m.geom_bodyid[other])
                    ) == root_body(body):
                        continue

                    force = sign * raw[j, :3]
                    rotation = xm[w, body].reshape(3, 3)
                    local = (pos[c] - xp[w, body]) @ rotation
                    weight = float(np.linalg.norm(force))
                    f[e, side, t] += force
                    fb[e, side, t] += force @ rotation
                    weighted[e, side, t] += weight * local
                    weights[e, side, t] += weight
                    self.last_totals[e, side] += force
                    self.last_records.append((w, e, side, t, g))
                    self.contact_records.append(dict(
                        env_id=e, side=("left", "right")[side], taxel=t,
                        force_world_N=force.tolist(),
                        force_ankle_N=(force @ rotation).tolist(),
                        position_ankle_m=local.tolist(),
                        other_geom=mj.mj_id2name(
                            m, mj.mjtObj.mjOBJ_GEOM, other
                        ) if other >= 0 else None,
                    ))

        location = np.full_like(f, np.nan)
        np.divide(
            weighted, weights[..., None], out=location,
            where=weights[..., None] > 1e-8
        )
        ids_t = torch.as_tensor(np.flatnonzero(selected), device=self.device)
        for name, array in (
            ("force_w", f), ("force_b", fb), ("contact_pos_b", location)
        ):
            getattr(self._data, name)[ids_t] = torch.as_tensor(
                array[selected], dtype=torch.float32, device=self.device
            )
        self._data.in_contact[ids_t] = torch.as_tensor(
            np.linalg.norm(f[selected], axis=-1) > self.cfg.force_threshold,
            device=self.device,
        )
        self._data.valid[ids_t] = torch.as_tensor(
            valid[selected], device=self.device
        )

@configclass
class TactileArrayCfg(SensorBaseCfg):
    class_type: type = TactileArray
    force_threshold: float = 1.0
