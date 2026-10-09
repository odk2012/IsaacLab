# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for clip alignment and independent motion boundaries; no simulator."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Load this dependency-light module directly, without registering unrelated tasks.
MODULE = Path(__file__).parents[1] / "isaaclab_tasks/contrib/mimic/mdp/motion_loader.py"
spec = importlib.util.spec_from_file_location("mimic_motion_loader", MODULE)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
MotionLibrary = loader.MotionLibrary
MotionLoader = loader.MotionLoader


class MotionLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def clip(self, name, frames, offset=0.0, reverse=False, metadata=True, **overrides):
        joints = ["left", "right"]
        bodies = ["pelvis", "foot"]
        pos = np.tile(np.array([offset + 1, offset + 2], dtype=np.float32), (frames, 1))
        body_pos = np.zeros((frames, 2, 3), dtype=np.float32)
        body_pos[:, 0, 0] = offset + 10
        body_pos[:, 1, 0] = offset + 20
        vel = np.zeros_like(body_pos)
        vel[:, :, 0] = offset + np.arange(frames)[:, None] / 60
        quat = np.zeros((frames, 2, 4), dtype=np.float32)
        quat[..., 3] = 1
        if reverse:
            joints.reverse()
            bodies.reverse()
            pos = pos[:, ::-1]
            body_pos = body_pos[:, ::-1]
            vel = vel[:, ::-1]
            quat = quat[:, ::-1]
        data = dict(
            fps=np.array([60]),
            joint_pos=pos,
            joint_vel=pos * 2,
            body_pos_w=body_pos,
            body_quat_w=quat,
            body_lin_vel_w=vel,
            body_ang_vel_w=vel * 2,
        )
        if metadata:
            data.update(joint_names=joints, body_names=bodies, quaternion_order="xyzw")
        else:
            data["body_quat_w"] = quat[..., [3, 0, 1, 2]]
        data.update(overrides)
        path = self.root / name
        np.savez(path, **data)
        return str(path)

    def library(self, files, bodies=None):
        return MotionLibrary(files, bodies, ["left", "right"], "cpu", 1 / 60)

    def test_mixed_clip_rows_are_aligned_by_names(self):
        a = self.clip("a.npz", 4)
        b = self.clip("b.npz", 7, offset=100, reverse=True)
        lib = self.library([a, b], ["foot", "pelvis"])
        rows = lib.frame_indices(torch.tensor([1, 0, 1]), torch.tensor([6, 0, 0]))
        self.assertEqual(rows.tolist(), [10, 0, 4])
        self.assertEqual(lib.lengths.tolist(), [4, 7])
        self.assertEqual(lib.joint_pos[rows].tolist(), [[101, 102], [1, 2], [101, 102]])
        self.assertEqual(lib.joint_vel[rows].tolist(), [[202, 204], [2, 4], [202, 204]])
        self.assertEqual(lib.body_pos_w[rows, :, 0].tolist(), [[120, 110], [20, 10], [120, 110]])
        self.assertEqual(lib.body_quat_w[rows, 0].tolist(), [[0, 0, 0, 1]] * 3)

    def test_acceleration_does_not_cross_clip_boundary(self):
        lib = self.library([self.clip("a.npz", 4), self.clip("b.npz", 7, offset=100)])
        linear, angular = lib.body_accelerations(0, 1 / 60)
        np.testing.assert_allclose(linear[:, 0], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], atol=5e-4)
        np.testing.assert_allclose(angular[:, 0], [2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0], atol=1e-3)
        self.assertEqual(linear[:, 1:].count_nonzero().item(), 0)

    def test_legacy_single_clip_matches_loader(self):
        path = self.clip("legacy.npz", 4, metadata=False)
        lib = self.library([path], ["pelvis", "foot"])
        legacy = MotionLoader(path, [0, 1])
        torch.testing.assert_close(lib.joint_pos, legacy.joint_pos)
        torch.testing.assert_close(lib.body_quat_w, legacy.body_quat_w)
        self.assertEqual(lib.body_quat_w[0, 0].tolist(), [0, 0, 0, 1])
        self.assertEqual(lib.time_step_total, 4)

    def test_incompatible_data_is_rejected(self):
        cases = [
            ({"fps": np.array([30])}, "control rate"),
            ({"joint_names": ["left", "other"]}, "joint names"),
            ({"body_names": ["pelvis", "pelvis"]}, "unique body names"),
            ({"joint_vel": np.zeros((3, 2))}, "joint_vel"),
            ({"body_quat_w": np.zeros((4, 2, 4))}, "unit norm"),
            ({"quaternion_order": "invalid"}, "quaternion order"),
        ]
        for index, (data, message) in enumerate(cases):
            with self.subTest(message=message):
                path = self.clip(f"bad{index}.npz", 4, **data)
                with self.assertRaisesRegex(ValueError, message):
                    self.library([path])

    def test_multiple_untagged_clips_require_joint_labels(self):
        files = [self.clip("a.npz", 4, metadata=False), self.clip("b.npz", 7, metadata=False)]
        with self.assertRaisesRegex(ValueError, "joint names"):
            self.library(files, ["pelvis", "foot"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
