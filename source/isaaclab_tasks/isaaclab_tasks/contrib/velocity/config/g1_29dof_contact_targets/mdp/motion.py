# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion clips, each on its own terrain, with the foot and hand contacts measured from the clip's geometry."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

CONTACT_BODIES = ("left_foot", "right_foot", "left_hand", "right_hand")
"""Order of the contact bodies in every contact array; mirroring swaps 0 with 1 and 2 with 3."""


def terrain_distance(data, points: np.ndarray) -> np.ndarray:
    """Signed distance [m] from ``points`` (..., 3) to a clip's ground plane and terrain boxes, in the clip frame."""
    flat = points.reshape(-1, 3)
    ground = float(data["ground_height"]) if "ground_height" in data else 0.0
    distance = flat[:, 2] - ground
    if "terrain_position" in data and len(data["terrain_position"]):
        positions, sizes = data["terrain_position"], data["terrain_dimension"]
        quats = data["terrain_quat"] if "terrain_quat" in data else np.tile((0.0, 0.0, 0.0, 1.0), (len(positions), 1))
        for center, size, quat in zip(positions, sizes, quats):
            local = Rotation.from_quat(quat).inv().apply(flat - center)
            excess = np.abs(local) - size / 2.0
            box = np.linalg.norm(np.maximum(excess, 0.0), axis=-1) + np.minimum(excess.max(axis=-1), 0.0)
            distance = np.minimum(distance, box)
    return distance.reshape(points.shape[:-1])


def clean_contacts(contact: np.ndarray, min_frames: int) -> np.ndarray:
    """Fill gaps and drop contacts shorter than ``min_frames`` frames in a (frames,) boolean array."""
    contact = contact.copy()
    for value in (False, True):  # fill short gaps, then drop short contacts
        start = 0
        while start < len(contact):
            end = start
            while end < len(contact) and contact[end] == contact[start]:
                end += 1
            inner = start > 0 and end < len(contact)
            if contact[start] == value and end - start < min_frames and (inner or value):
                contact[start:end] = not value
            start = end
    return contact


def upcoming_contacts(
    contact: np.ndarray, positions: np.ndarray, fps: float, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Target position and time to contact of the current or next contact, for every frame of one body.

    The target of a contact is the mean body position over it. Frames with no contact left in the clip get the
    body's own position and the time ``horizon`` [s]; times are capped at ``horizon``.
    """
    num = len(contact)
    target = positions.copy()
    time = np.full(num, horizon)
    start = 0
    segments = []
    while start < num:
        end = start
        while end < num and contact[end] == contact[start]:
            end += 1
        if contact[start]:
            segments.append((start, end))
        start = end
    for seg_start, seg_end in reversed(segments):
        mean = positions[seg_start:seg_end].mean(axis=0)
        # this contact is the next one for every frame from the end of the previous contact to its own end
        first = max(0, seg_start - int(horizon * fps))
        target[first:seg_end] = mean
        time[first:seg_end] = np.clip((seg_start - np.arange(first, seg_end)) / fps, 0.0, horizon)
    return target, time


class ContactMotionLibrary:
    """Reference clips packed in articulation joint order, with their foot and hand contacts.

    Every clip keeps its own terrain, built by the environment on its own terrain column. Contacts are measured
    from the clip's geometry: a foot is in contact when at least two corners of its foot box are within
    ``contact_margin`` of the terrain and its ankle moves slower than ``contact_speed``, a hand when its sphere is
    within ``contact_margin`` and moves slower than ``contact_speed``. Contacts and gaps shorter than
    ``min_contact_frames`` are dropped. For each frame and :data:`CONTACT_BODIES` entry, the library stores
    the position of the current or next contact (the foot's ankle roll link, or the hand sphere center), whether
    the body is in contact, and the time until the contact begins.
    """

    def __init__(
        self,
        motion_files: Sequence[str],
        joint_names: Sequence[str],
        root_body_name: str,
        foot_body_names: Sequence[str],
        step_dt: float,
        device: str,
        contact_margin: float = 0.01,
        contact_speed: float = 0.3,
        min_contact_frames: int = 3,
        horizon: float = 2.0,
    ):
        if not motion_files:
            raise ValueError("At least one motion file is required.")
        keys = ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "body_pos")
        arrays = {key: [] for key in (*keys, "target", "contact", "time")}
        lengths = []
        for path in motion_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Motion file not found: {path}")
            with np.load(path, allow_pickle=False) as data:
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if not math.isclose(fps * step_dt, 1.0, rel_tol=1.0e-6):
                    raise ValueError(f"{path}: motion FPS {fps} must match the control rate {1.0 / step_dt} Hz.")
                if "quaternion_order" in data and str(np.asarray(data["quaternion_order"]).item()) != "xyzw":
                    raise ValueError(f"{path}: expected xyzw body quaternions.")
                clip_joints = data["joint_names"].tolist()
                clip_bodies = data["body_names"].tolist()
                if set(clip_joints) != set(joint_names):
                    raise ValueError(f"{path}: motion and robot joint names must cover the same joints.")
                joints = [clip_joints.index(name) for name in joint_names]
                root = clip_bodies.index(root_body_name)
                feet = [clip_bodies.index(name) for name in foot_body_names]
                arrays["joint_pos"].append(data["joint_pos"][:, joints])
                arrays["joint_vel"].append(data["joint_vel"][:, joints])
                arrays["root_pos"].append(data["body_pos_w"][:, root])
                arrays["root_quat"].append(data["body_quat_w"][:, root])
                arrays["root_lin_vel"].append(data["body_lin_vel_w"][:, root])
                arrays["root_ang_vel"].append(data["body_ang_vel_w"][:, root])
                # contact bodies: the feet's ankle roll links and the hand spheres
                frames = data["joint_pos"].shape[0]
                corners = data["foot_box_corners_w"].reshape(frames, 2, -1, 3)
                hands = data["hand_sphere_centers_w"].reshape(frames, 2, -1, 3)[:, :, 0]
                radii = np.broadcast_to(np.asarray(data["hand_sphere_radii"], dtype=float).reshape(-1), (2,))
                body_pos = np.concatenate((data["body_pos_w"][:, feet], hands), axis=1)
                # a flat, planted foot: two corners on the terrain and a slow ankle; dragged swing feet are not
                foot_touch = (terrain_distance(data, corners) < contact_margin).sum(axis=-1) >= 2
                hand_touch = terrain_distance(data, hands) - radii < contact_margin
                speed = np.linalg.norm(np.gradient(body_pos, axis=0) * fps, axis=-1)
                touch = np.concatenate((foot_touch, hand_touch), axis=1) & (speed < contact_speed)
                contact = np.stack([clean_contacts(touch[:, i], min_contact_frames) for i in range(4)], axis=1)
                targets, times = zip(
                    *(upcoming_contacts(contact[:, i], body_pos[:, i], fps, horizon) for i in range(4))
                )
                arrays["body_pos"].append(body_pos)
                arrays["target"].append(np.stack(targets, axis=1))
                arrays["contact"].append(contact)
                arrays["time"].append(np.stack(times, axis=1))
                lengths.append(frames)

        def pack(key: str, dtype=torch.float32) -> torch.Tensor:
            return torch.tensor(np.concatenate(arrays[key]), dtype=dtype, device=device)

        self.fps = fps
        self.horizon = horizon
        self.joint_pos = pack("joint_pos")
        """Joint positions [rad] in articulation order. Shape is (frames, joints)."""
        self.joint_vel = pack("joint_vel")
        """Joint velocities [rad/s] in articulation order. Shape is (frames, joints)."""
        self.root_pos = pack("root_pos")
        """Root positions [m] in the clip frame. Shape is (frames, 3)."""
        self.root_quat = pack("root_quat")
        """Root orientations (x, y, z, w) in the clip frame. Shape is (frames, 4)."""
        self.root_lin_vel = pack("root_lin_vel")
        """Root linear velocities [m/s] in the clip frame. Shape is (frames, 3)."""
        self.root_ang_vel = pack("root_ang_vel")
        """Root angular velocities [rad/s] in the clip frame. Shape is (frames, 3)."""
        self.body_pos = pack("body_pos")
        """Contact body positions [m] in the clip frame. Shape is (frames, 4, 3)."""
        self.contact_target = pack("target")
        """Position [m] of each contact body's current or next contact, in the clip frame. Shape is (frames, 4, 3)."""
        self.in_contact = pack("contact", torch.bool)
        """Whether each contact body is in contact. Shape is (frames, 4)."""
        self.time_to_contact = pack("time")
        """Time [s] until each body's next contact begins, 0 in contact, capped at the horizon. Shape (frames, 4)."""
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.starts = self.lengths.cumsum(0) - self.lengths

    def frame_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Map clip IDs and clip-local frame numbers to packed rows."""
        return self.starts[motion_ids] + time_steps
