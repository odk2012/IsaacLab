
"""GPU aggregation prototype; final physics-substep contact samples.

Geometry binding runs on CPU once. Warm updates and resets use GPU tensors.
CPU debug contact-record lists are not provided by this class.
"""
import inspect
import re
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass
from .tactile_array import TactileArray, TactileArrayCfg


def check_contact_api():
    module = inspect.getmodule(mw.contact_force)
    kernel = getattr(module, "contact_force_kernel", None)
    source = inspect.getsource(getattr(kernel, "func", kernel))
    assert re.search(
        r"if\s+contactid\s*>=\s*nacon(?:_in)?\[0\]:\s*return", source
    ), "Installed contact_force needs a reviewed inactive-contact guard."


class TactileArrayGPU(TactileArray):
    def _scope(self):
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        return wp.ScopedStream(stream, sync_enter=True, sync_exit=True)

    def _initialize_impl(self):
        super()._initialize_impl()
        self._gpu_ready = False
        self._skip = torch.ones(
            self._num_envs, dtype=torch.bool, device=self.device
        )
        self.last_totals = self.last_records = self.contact_records = None

    def _setup_gpu(self):
        check_contact_api()
        self._bind_solver()
        m, d = self.solver.mj_model, self.solver.mjw_data
        self._nw, self._ng = self.slot.shape[:2]

        channel = (
            self.slot[..., 0]*54
            + self.slot[..., 1]*27
            + self.slot[..., 2]
        )
        channel[self.slot[..., 0] < 0] = -1
        self._channel = torch.as_tensor(
            channel, device=self.device, dtype=torch.long
        )
        self._body = torch.as_tensor(
            m.geom_bodyid.copy(), device=self.device, dtype=torch.long
        )
        roots = np.arange(m.nbody)
        for b in range(m.nbody):
            while m.body_parentid[roots[b]] != 0:
                roots[b] = m.body_parentid[roots[b]]
        self._roots = torch.as_tensor(
            roots, device=self.device, dtype=torch.long
        )

        self._ids = torch.arange(
            d.naconmax, device=self.device, dtype=torch.int32
        )
        self._wp_ids = wp.from_torch(self._ids, dtype=wp.int32)
        self._raw = torch.zeros((d.naconmax, 6), device=self.device)
        self._wp_raw = wp.from_torch(
            self._raw, dtype=wp.spatial_vector
        )
        self._sum = torch.zeros(
            (self._num_envs*54, 10), device=self.device
        )
        self.health = torch.zeros(
            3, dtype=torch.long, device=self.device
        )
        self._gpu_ready = True

    def reset(self, env_ids=None, env_mask=None):
        if not hasattr(self, "_data"):
            return SensorBase.reset(
                self, env_ids=env_ids, env_mask=env_mask
            )
        with self._scope():
            mask = self._resolve_indices_and_mask(env_ids, env_mask)
            SensorBase.reset(self, env_mask=mask)
            selected = wp.to_torch(mask)
            self._skip.logical_or_(selected)
            for field in ("force_w", "force_b"):
                getattr(self._data, field).masked_fill_(
                    selected[:, None, None, None], 0
                )
            self._data.contact_pos_b.masked_fill_(
                selected[:, None, None, None], float("nan")
            )
            self._data.in_contact.masked_fill_(
                selected[:, None, None], False
            )
            self._data.valid.masked_fill_(selected, False)

    def _update_buffers_impl(self, env_mask):
        with self._scope():
            if not self._gpu_ready:
                self._setup_gpu()
            s, data = self.solver, self._data
            d = s.mjw_data
            selected = wp.to_torch(env_mask)
            valid = selected & ~self._skip
            self._skip.logical_and_(~selected)

            self._raw.zero_()
            mw.contact_force(
                s.mjw_model, d, self._wp_ids, True, self._wp_raw
            )
            count = wp.to_torch(d.nacon)[0]
            active = self._ids < count
            worlds = wp.to_torch(d.contact.worldid).long()
            pairs = wp.to_torch(d.contact.geom).long()
            bad_world = active & (
                (worlds < 0) | (worlds >= self._nw)
            )
            bad_geom = active & (
                (pairs < -1) | (pairs >= self._ng)
            ).any(dim=1)
            self.health.copy_(torch.stack((
                ((count < 0) | (count >= d.naconmax)).long(),
                (bad_world | bad_geom).sum(),
                (active & ~torch.isfinite(self._raw).all(dim=1)).sum(),
            )))

            w = worlds.clamp(0, self._nw-1)
            p = wp.to_torch(d.contact.pos)
            xp = wp.to_torch(d.xpos)
            xm = wp.to_torch(d.xmat).reshape(self._nw, -1, 3, 3)
            self._sum.zero_()

            for endpoint, sign in ((0, -1.0), (1, 1.0)):
                geom = pairs[:, endpoint]
                other = pairs[:, 1-endpoint]
                g = geom.clamp(0, self._ng-1)
                body = self._body[g]
                other_body = self._body[other.clamp(0, self._ng-1)]
                channel = self._channel[w, g]
                safe_channel = channel.clamp_min(0)
                env = safe_channel // 54
                self_contact = (other >= 0) & (
                    self._roots[body] == self._roots[other_body]
                )
                take = (
                    active & ~bad_world & ~bad_geom
                    & (geom >= 0) & (channel >= 0)
                    & valid[env] & ~self_contact
                )
                rotation = xm[w, body].transpose(1, 2)
                force = torch.where(
                    take[:, None], sign*self._raw[:, :3], 0.0
                )
                force_b = torch.bmm(
                    rotation, force.unsqueeze(2)
                ).squeeze(2)
                local = torch.bmm(
                    rotation, (p-xp[w, body]).unsqueeze(2)
                ).squeeze(2)
                local = torch.where(take[:, None], local, 0.0)
                force_b = torch.where(take[:, None], force_b, 0.0)
                weight = torch.linalg.vector_norm(
                    force, dim=1, keepdim=True
                )
                values = torch.cat(
                    (force, force_b, local*weight, weight), dim=1
                )
                self._sum.index_add_(0, safe_channel, values)

            total = self._sum.reshape(self._num_envs, 2, 27, 10)
            weight = total[..., 9:10]
            pos = torch.where(
                weight > 1e-8,
                total[..., 6:9]/weight.clamp_min(1e-8),
                float("nan"),
            )
            mask = selected[:, None, None, None]
            for name, value in (
                ("force_w", total[..., :3]),
                ("force_b", total[..., 3:6]),
                ("contact_pos_b", pos),
            ):
                target = getattr(data, name)
                target.copy_(torch.where(mask, value, target))
            loaded = (
                torch.linalg.vector_norm(total[..., :3], dim=-1)
                > self.cfg.force_threshold
            )
            data.in_contact.copy_(torch.where(
                selected[:, None, None], loaded, data.in_contact
            ))
            data.valid.copy_(torch.where(selected, valid, data.valid))


@configclass
class TactileArrayGPUCfg(TactileArrayCfg):
    class_type: type = TactileArrayGPU
