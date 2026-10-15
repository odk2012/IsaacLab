# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and align finite reference clips for humanoid motion tracking."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        quaternion_order: Literal["wxyz", "xyzw"] = "wxyz",
        source_joint_names: Sequence[str] | None = None,
        target_joint_names: Sequence[str] | None = None,
    ):
        """Load motion with runtime quaternions in xyzw order.

        NPZ ``quaternion_order`` metadata takes precedence over the fallback argument.
        Untagged legacy motion files default to wxyz; untagged xyzw files must opt in.
        Explicit source/target joint names reorder both joint position and velocity.
        Omitting both preserves the stored joint column order.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file, allow_pickle=False)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        if "quaternion_order" in data:
            quaternion_order = str(np.asarray(data["quaternion_order"]).item())
        if quaternion_order == "wxyz":
            self._body_quat_w = self._body_quat_w.roll(-1, dims=-1)
        elif quaternion_order != "xyzw":
            raise ValueError(f"Unsupported quaternion order: {quaternion_order}")
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        data.close()
        if (source_joint_names is None) != (target_joint_names is None):
            raise ValueError("Both source_joint_names and target_joint_names must be provided together.")
        if source_joint_names is not None:
            source_names = list(source_joint_names)
            target_names = list(target_joint_names)
            if self.joint_pos.ndim != 2 or self.joint_vel.shape != self.joint_pos.shape:
                raise ValueError("Motion joint position and velocity must have matching [frames, joints] shapes.")
            if (
                len(source_names) != self.joint_pos.shape[1]
                or len(set(source_names)) != len(source_names)
                or len(set(target_names)) != len(target_names)
                or set(source_names) != set(target_names)
            ):
                raise ValueError("Motion and robot joint names must uniquely cover the same joint columns.")
            joint_indexes = torch.tensor([source_names.index(name) for name in target_names], device=device)
            self.joint_pos = self.joint_pos[:, joint_indexes]
            self.joint_vel = self.joint_vel[:, joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionLibrary(MotionLoader):
    """Pack clips in a common joint/body order without crossing their boundaries.

    Frame indices in the packed arrays are internal storage indices. Episode clocks
    remain local to each clip; use ``frame_indices`` to look up reference values.
    All clips must use the control rate: this class does not resample input data.
    """

    def __init__(
        self,
        motion_files: Sequence[str],
        body_names: Sequence[str] | None,
        robot_joint_names: Sequence[str],
        device: str,
        step_dt: float,
        quaternion_order: Literal["wxyz", "xyzw"] = "wxyz",
        fallback_joint_names: Sequence[str] | None = None,
    ):
        if not motion_files:
            raise ValueError("At least one motion file is required.")
        self.motion_files = tuple(motion_files)
        self.body_names = None
        self.tracked_body_names = list(body_names) if body_names is not None else None
        clips = []
        for path in motion_files:
            with np.load(path, allow_pickle=False) as data:
                joints = data["joint_names"].tolist() if "joint_names" in data else fallback_joint_names
                bodies = data["body_names"].tolist() if "body_names" in data else body_names
                if bodies is None or len(set(bodies)) != len(bodies):
                    raise ValueError(f"{path}: provide unique body names in metadata or configuration.")
                bodies = list(bodies)
                if self.body_names is None:
                    self.body_names = bodies
                if set(bodies) != set(self.body_names):
                    raise ValueError(f"{path}: clips must contain the same named bodies.")
                if self.tracked_body_names is None:
                    self.tracked_body_names = bodies
                if len(set(self.tracked_body_names)) != len(self.tracked_body_names):
                    raise ValueError("Tracked body names must be unique.")
                if not set(self.tracked_body_names).issubset(bodies):
                    raise ValueError(f"{path}: tracked bodies are missing.")
                if len(motion_files) > 1 and joints is None:
                    raise ValueError(f"{path}: multiple clips require joint names or motion_joint_names.")
                self._validate_arrays(data, path, len(robot_joint_names), len(bodies), step_dt)
            clip = MotionLoader(
                path,
                list(range(len(bodies))),
                device,
                quaternion_order,
                joints,
                robot_joint_names if joints is not None else None,
            )
            # Every clip uses the first clip's full body order, including the root
            # and any assist body that is not selected for tracking rewards.
            order = [bodies.index(name) for name in self.body_names]
            clip._body_pos_w = clip._body_pos_w[:, order]
            clip._body_quat_w = clip._body_quat_w[:, order]
            clip._body_lin_vel_w = clip._body_lin_vel_w[:, order]
            clip._body_ang_vel_w = clip._body_ang_vel_w[:, order]
            clips.append(clip)
        self.fps = clips[0].fps
        self.lengths = torch.tensor([clip.time_step_total for clip in clips], dtype=torch.long, device=device)
        self.starts = self.lengths.cumsum(0) - self.lengths
        self.ends = self.starts + self.lengths - 1
        self.frame_ranges = []
        start = 0
        for clip in clips:
            self.frame_ranges.append((start, start + clip.time_step_total))
            start += clip.time_step_total
        self.time_step_total = start
        self._body_indexes = [self.body_names.index(name) for name in self.tracked_body_names]
        self.joint_pos = torch.cat([clip.joint_pos for clip in clips])
        self.joint_vel = torch.cat([clip.joint_vel for clip in clips])
        self._body_pos_w = torch.cat([clip._body_pos_w for clip in clips])
        self._body_quat_w = torch.cat([clip._body_quat_w for clip in clips])
        self._body_lin_vel_w = torch.cat([clip._body_lin_vel_w for clip in clips])
        self._body_ang_vel_w = torch.cat([clip._body_ang_vel_w for clip in clips])

    def frame_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Map per-environment clip IDs and local frame numbers to packed rows."""
        return self.starts[motion_ids] + time_steps

    def body_accelerations(self, body_index: int, step_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """World linear/angular acceleration, with each clip's last frame zeroed."""
        lin_vel = self._body_lin_vel_w[:, body_index]
        ang_vel = self._body_ang_vel_w[:, body_index]
        lin_acc = torch.zeros_like(lin_vel)
        ang_acc = torch.zeros_like(ang_vel)
        lin_acc[:-1] = (lin_vel[1:] - lin_vel[:-1]) / step_dt
        ang_acc[:-1] = (ang_vel[1:] - ang_vel[:-1]) / step_dt
        lin_acc[self.ends] = 0
        ang_acc[self.ends] = 0
        return lin_acc, ang_acc

    @staticmethod
    def _validate_arrays(data, path: str, num_joints: int, num_bodies: int, step_dt: float) -> None:
        fps_data = np.asarray(data["fps"])
        if fps_data.size != 1:
            raise ValueError(f"{path}: fps must contain exactly one value.")
        fps = float(fps_data.reshape(-1)[0])
        if not math.isfinite(fps) or fps <= 0 or not math.isclose(fps * step_dt, 1.0, abs_tol=1e-5):
            raise ValueError(f"{path}: motion fps {fps} must match control rate {1 / step_dt}.")
        frames = data["joint_pos"].shape[0]
        if frames < 2:
            raise ValueError(f"{path}: a clip must have at least two frames.")
        shapes = {
            "joint_pos": (frames, num_joints),
            "joint_vel": (frames, num_joints),
            "body_pos_w": (frames, num_bodies, 3),
            "body_quat_w": (frames, num_bodies, 4),
            "body_lin_vel_w": (frames, num_bodies, 3),
            "body_ang_vel_w": (frames, num_bodies, 3),
        }
        for key, shape in shapes.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"{path}: {key} must be finite with shape {shape}.")
        norms = np.linalg.norm(data["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(f"{path}: body quaternions must have unit norm.")
