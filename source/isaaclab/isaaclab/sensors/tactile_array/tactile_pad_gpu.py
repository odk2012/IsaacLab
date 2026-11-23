import re
import numpy as np
import torch
import warp as wp
import mujoco as mj
import mujoco_warp as mw
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager
from isaaclab.sensors.tactile_array.tactile_array import TactileArrayData
from isaaclab.sensors.tactile_array.tactile_array_gpu import (
    TactileArrayGPU, TactileArrayGPUCfg, check_contact_api
)

class TactilePadGPU(TactileArrayGPU):
    """102 physical box taxels per foot; no spatial interpolation."""

    def _initialize_impl(self):
        SensorBase._initialize_impl(self)
        self._allocate_buffers()

    def _allocate_buffers(self):
        shape = (self._num_envs, 2, 102, 3)
        self._data = TactileArrayData(
            torch.zeros(shape, device=self.device),
            torch.zeros(shape, device=self.device),
            torch.full(shape, float("nan"), device=self.device),
            torch.full(shape, float("nan"), device=self.device),
            torch.zeros(shape[:-1], dtype=torch.bool, device=self.device),
            torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
        )
        self._skip = torch.ones(
            self._num_envs, dtype=torch.bool, device=self.device
        )
        self._gpu_ready = False
        self.solver = None
        self.bindings = []
        self.world_env = {}
        self.last_totals = self.last_records = self.contact_records = None

    def _bind_solver(self):
        self.solver = NewtonManager._solver
        s = self.solver
        m, d = s.mj_model, s.mjw_data
        mapping = s.mjc_geom_to_newton_shape.numpy()
        labels = getattr(s.model, "shape_label", None)
        if labels is None:
            labels = s.model.shape_key
        nw, ng = mapping.shape
        assert nw == self._num_envs == d.xpos.shape[0]

        self.slot = np.full((nw, ng, 3), -1, dtype=np.int32)
        centers = np.full((self._num_envs, 2, 102, 3), np.nan)
        seen = set()
        self.bindings = []
        self.world_env = {}

        for w in range(nw):
            for g in range(ng):
                sh = int(mapping[w, g])
                if sh < 0:
                    continue
                name = labels[sh]
                match = re.search(r"/tactile_pad_(\d{3})(?:_\d+)?$", name)
                if match is None:
                    continue
                env = re.search(r"/env_(\d+)/", name)
                assert env is not None, name
                e, t = int(env.group(1)), int(match.group(1))
                side = 0 if "/left_ankle_roll_link/" in name else 1
                expected_link = (
                    "/left_ankle_roll_link/" if side == 0
                    else "/right_ankle_roll_link/"
                )
                assert expected_link in name
                assert 0 <= e < self._num_envs and 0 <= t < 102
                assert (e, side, t) not in seen
                assert m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX
                assert np.allclose(
                    m.geom_size[g], [.00495, .00495, .0025],
                    atol=1e-7, rtol=0,
                )
                assert m.geom_bodyid[g] > 0
                assert self.world_env.setdefault(w, e) == e

                seen.add((e, side, t))
                self.slot[w, g] = [e, side, t]
                centers[e, side, t] = m.geom_pos[g]
                self.bindings.append(dict(
                    world=w, env=e, geom=g, shape=sh,
                    side=side, taxel=t, name=name,
                ))

        assert len(seen) == self._num_envs * 204, (
            "Incomplete pad bindings", len(seen)
        )
        assert set(self.world_env.values()) == set(range(self._num_envs))
        self._data.taxel_pos_b.copy_(torch.as_tensor(
            centers, dtype=torch.float32, device=self.device
        ))
        print("PAD_SENSOR_BOUND:", len(seen), flush=True)

    def _setup_gpu(self):
        check_contact_api()
        self._bind_solver()
        m, d = (self.solver.mj_model, self.solver.mjw_data)
        self._nw, self._ng = self.slot.shape[:2]
        channel = self.slot[..., 0] * 204 + self.slot[..., 1] * 102 + self.slot[..., 2]
        channel[self.slot[..., 0] < 0] = -1
        self._channel = torch.as_tensor(channel, device=self.device, dtype=torch.long)
        self._body = torch.as_tensor(m.geom_bodyid.copy(), device=self.device, dtype=torch.long)
        roots = np.arange(m.nbody)
        for b in range(m.nbody):
            while m.body_parentid[roots[b]] != 0:
                roots[b] = m.body_parentid[roots[b]]
        self._roots = torch.as_tensor(roots, device=self.device, dtype=torch.long)
        self._ids = torch.arange(d.naconmax, device=self.device, dtype=torch.int32)
        self._wp_ids = wp.from_torch(self._ids, dtype=wp.int32)
        self._raw = torch.zeros((d.naconmax, 6), device=self.device)
        self._wp_raw = wp.from_torch(self._raw, dtype=wp.spatial_vector)
        self._sum = torch.zeros((self._num_envs * 204, 10), device=self.device)
        self.health = torch.zeros(3, dtype=torch.long, device=self.device)
        self._gpu_ready = True

    def _update_buffers_impl(self, env_mask):
        with self._scope():
            if not self._gpu_ready:
                self._setup_gpu()
            s, data = (self.solver, self._data)
            d = s.mjw_data
            selected = wp.to_torch(env_mask)
            valid = selected & ~self._skip
            self._skip.logical_and_(~selected)
            self._raw.zero_()
            mw.contact_force(s.mjw_model, d, self._wp_ids, True, self._wp_raw)
            count = wp.to_torch(d.nacon)[0]
            active = self._ids < count
            worlds = wp.to_torch(d.contact.worldid).long()
            pairs = wp.to_torch(d.contact.geom).long()
            bad_world = active & ((worlds < 0) | (worlds >= self._nw))
            bad_geom = active & ((pairs < -1) | (pairs >= self._ng)).any(dim=1)
            self.health.copy_(torch.stack((((count < 0) | (count >= d.naconmax)).long(), (bad_world | bad_geom).sum(), (active & ~torch.isfinite(self._raw).all(dim=1)).sum())))
            w = worlds.clamp(0, self._nw - 1)
            p = wp.to_torch(d.contact.pos)
            xp = wp.to_torch(d.xpos)
            xm = wp.to_torch(d.xmat).reshape(self._nw, -1, 3, 3)
            self._sum.zero_()
            for endpoint, sign in ((0, -1.0), (1, 1.0)):
                geom = pairs[:, endpoint]
                other = pairs[:, 1 - endpoint]
                g = geom.clamp(0, self._ng - 1)
                body = self._body[g]
                other_body = self._body[other.clamp(0, self._ng - 1)]
                channel = self._channel[w, g]
                safe_channel = channel.clamp_min(0)
                env = safe_channel // 204
                self_contact = (other >= 0) & (self._roots[body] == self._roots[other_body])
                take = active & ~bad_world & ~bad_geom & (geom >= 0) & (channel >= 0) & valid[env] & ~self_contact
                rotation = xm[w, body].transpose(1, 2)
                force = torch.where(take[:, None], sign * self._raw[:, :3], 0.0)
                force_b = torch.bmm(rotation, force.unsqueeze(2)).squeeze(2)
                local = torch.bmm(rotation, (p - xp[w, body]).unsqueeze(2)).squeeze(2)
                local = torch.where(take[:, None], local, 0.0)
                force_b = torch.where(take[:, None], force_b, 0.0)
                weight = torch.linalg.vector_norm(force, dim=1, keepdim=True)
                values = torch.cat((force, force_b, local * weight, weight), dim=1)
                self._sum.index_add_(0, safe_channel, values)
            total = self._sum.reshape(self._num_envs, 2, 102, 10)
            weight = total[..., 9:10]
            pos = torch.where(weight > 1e-08, total[..., 6:9] / weight.clamp_min(1e-08), float('nan'))
            mask = selected[:, None, None, None]
            for name, value in (('force_w', total[..., :3]), ('force_b', total[..., 3:6]), ('contact_pos_b', pos)):
                target = getattr(data, name)
                target.copy_(torch.where(mask, value, target))
            loaded = torch.linalg.vector_norm(total[..., :3], dim=-1) > self.cfg.force_threshold
            data.in_contact.copy_(torch.where(selected[:, None, None], loaded, data.in_contact))
            data.valid.copy_(torch.where(selected, valid, data.valid))

@configclass
class TactilePadGPUCfg(TactileArrayGPUCfg):
    class_type: type = TactilePadGPU
