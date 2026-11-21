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


def contact_segments(contact: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) frames of each contact in a (frames,) boolean array."""
    segments = []
    start = 0
    while start < len(contact):
        end = start
        while end < len(contact) and contact[end] == contact[start]:
            end += 1
        if contact[start]:
            segments.append((start, end))
        start = end
    return segments


def contact_targets(contact: np.ndarray, positions: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    """Target position of one body for every frame: the contact it is in, else its next contact.

    The target of a contact is the mean body position over it. After lift-off the body keeps that contact as its
    target until ``other`` (the opposite foot's contact flags) is in contact, so one foot gets its new target only
    once the other foot has landed; without ``other``, the target moves on at lift-off. After its last contact the
    body keeps that contact; a body that never touches the terrain gets its own position.
    """
    target = positions.copy()
    segments = contact_segments(contact)
    if not segments:
        return target
    switch = 0
    for k, (seg_start, seg_end) in enumerate(segments):
        mean = positions[seg_start:seg_end].mean(axis=0)
        target[switch:seg_end] = mean
        switch = seg_end
        if other is not None:
            # hold this contact until the other foot is down (or this foot lands again)
            limit = segments[k + 1][0] if k + 1 < len(segments) else len(contact)
            while switch < limit and not other[switch]:
                switch += 1
            if switch == len(contact):
                switch = seg_end  # the other foot never lands again: move on at lift-off
            target[seg_end:switch] = mean
    target[switch:] = target[switch - 1]
    return target


class ContactMotionLibrary:
    """Reference clips packed in articulation joint order, with their foot and hand contacts.

    Every clip keeps its own terrain, built by the environment on its own terrain column. Contacts are measured
    from the clip's geometry: a foot is in contact when at least two corners of its foot box are within
    ``contact_margin`` of the terrain and its ankle moves slower than ``contact_speed``, a hand when its sphere is
    within ``contact_margin`` and moves slower than ``contact_speed``. Contacts and gaps shorter than
    ``min_contact_frames`` are dropped. For each frame and :data:`CONTACT_BODIES` entry, the library stores
    the position of its contact target (the foot's ankle roll link, or the hand sphere center; see
    :func:`contact_targets`): a foot keeps its last contact until the other foot lands, then targets its next
    contact. It also stores whether the body is in contact.
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
    ):
        if not motion_files:
            raise ValueError("At least one motion file is required.")
        keys = ("joint_pos", "joint_vel", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "body_pos")
        arrays = {key: [] for key in (*keys, "target", "contact", "touchdown", "next_touchdown")}
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
                # a foot keeps its contact until the other foot lands; hands move on at lift-off
                others = (contact[:, 1], contact[:, 0], None, None)
                targets = [contact_targets(contact[:, i], body_pos[:, i], others[i]) for i in range(4)]
                arrays["body_pos"].append(body_pos)
                arrays["target"].append(np.stack(targets, axis=1))
                arrays["contact"].append(contact)
                # touchdowns (a contact starting after the first frame), and each frame's next one (-1 when none)
                touchdown = np.zeros_like(contact)
                touchdown[1:] = contact[1:] & ~contact[:-1]
                next_touchdown = np.full(contact.shape, -1, dtype=np.int64)
                upcoming = np.full(4, -1, dtype=np.int64)
                for frame in range(frames - 1, -1, -1):
                    next_touchdown[frame] = upcoming
                    upcoming = np.where(touchdown[frame], frame, upcoming)
                arrays["touchdown"].append(touchdown)
                arrays["next_touchdown"].append(next_touchdown)
                lengths.append(frames)

        def pack(key: str, dtype=torch.float32) -> torch.Tensor:
            return torch.tensor(np.concatenate(arrays[key]), dtype=dtype, device=device)

        self.fps = fps
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
        self.touchdown = pack("touchdown", torch.bool)
        """Whether each contact body's contact starts at this frame (not a clip's first frame). Shape (frames, 4)."""
        self.next_touchdown = pack("next_touchdown", torch.long)
        """Clip-local frame of each contact body's next touchdown after this frame, -1 when none. Shape (frames, 4)."""
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.starts = self.lengths.cumsum(0) - self.lengths

    def frame_indices(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Map clip IDs and clip-local frame numbers to packed rows."""
        return self.starts[motion_ids] + time_steps
