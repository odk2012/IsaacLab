# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU reference-command contracts with a small robot descriptor, no physics."""

import importlib
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

# Import the real production module without eagerly registering unrelated tasks.
package = types.ModuleType("_mimic_command_contract")
package.__path__ = [str(Path(__file__).parents[1] / "isaaclab_tasks/contrib/mimic/mdp")]
sys.modules[package.__name__] = package
commands = importlib.import_module(package.__name__ + ".commands")


class MotionCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.files = []
        for i, frames in enumerate([120, 240, 180, 300]):
            path = Path(self.temp.name) / f"clip{i}.npz"
            quat = np.zeros((frames, 1, 4), dtype=np.float32)
            quat[..., 3] = 1
            np.savez(
                path,
                fps=[60],
                joint_names=["joint"],
                body_names=["pelvis"],
                quaternion_order="xyzw",
                joint_pos=np.full((frames, 1), i, dtype=np.float32),
                joint_vel=np.zeros((frames, 1), dtype=np.float32),
                body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
                body_quat_w=quat,
                body_lin_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
            )
            self.files.append(str(path))

    def command(self, **overrides):
        n = 64
        quat = torch.zeros(n, 1, 4)
        quat[..., 3] = 1
        # Describe the robot and current measured pose; production code owns all
        # sampling, clocks, remapping, reward gates and reference calculations.
        robot = SimpleNamespace(
            body_names=["pelvis"],
            joint_names=["joint"],
            find_bodies=lambda names, preserve_order: ([0], ["pelvis"]),
            data=SimpleNamespace(body_pos_w=torch.zeros(n, 1, 3), body_quat_w=quat),
        )

        class Scene(dict):
            pass

        scene = Scene(robot=robot)
        scene.env_origins = torch.zeros(n, 3)
        env = SimpleNamespace(
            scene=scene,
            num_envs=n,
            device="cpu",
            step_dt=1 / 60,
            sim=SimpleNamespace(vis_marker_registry=Mock()),
            termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool)),
        )
        values = dict(
            asset_name="robot",
            motion_files=self.files[:2],
            body_names=["pelvis"],
            anchor_body_name="pelvis",
            debug_vis=False,
            resampling_time_range=(1e9, 1e9),
            motion_stance_phase_ranges=[[], []],
            adaptive_kernel_size=3,
        )
        values.update(overrides)
        return commands.MotionCommand(commands.MotionCommandCfg(**values), env)

    def test_reset_draw_is_once_and_partial_reset_preserves_other_envs(self):
        torch.manual_seed(42)
        cmd = self.command()
        cmd.resample_time_steps(slice(None))
        before_ids, before_frames = cmd.motion_ids.clone(), cmd.time_steps.clone()
        cmd.resample_time_steps(torch.arange(cmd.num_envs))
        torch.testing.assert_close(cmd.motion_ids, before_ids)
        torch.testing.assert_close(cmd.time_steps, before_frames)
        self.assertEqual(set(cmd.motion_ids.tolist()), {0, 1})
        torch.testing.assert_close(cmd.joint_pos[:, 0], cmd.motion_ids.float())
        self.assertTrue(bool((cmd.time_steps < cmd.motion_lengths).all()))
        cmd._resample_command(slice(None))
        cmd.resample_time_steps(slice(0, 8))
        torch.testing.assert_close(cmd.motion_ids[8:], before_ids[8:])
        torch.testing.assert_close(cmd.time_steps[8:], before_frames[8:])

    def test_unequal_clips_stop_at_own_end_without_switching(self):
        cmd = self.command()
        cmd.motion_ids[1::2] = 1
        cmd.time_steps[:] = cmd.motion_lengths - 2
        ids = cmd.motion_ids.clone()
        cmd._update_command()
        self.assertTrue(bool(cmd.has_reached_end.all()))
        torch.testing.assert_close(cmd.phase, torch.ones(cmd.num_envs))
        self.assertEqual(cmd.reference_frame_indices[:2].tolist(), [119, 359])
        cmd._update_command()
        self.assertEqual(cmd.time_steps[:2].tolist(), [119, 239])
        torch.testing.assert_close(cmd.motion_ids, ids)

    def test_failure_and_assistance_stay_with_old_clip(self):
        torch.manual_seed(3)
        cmd = self.command()
        cmd.motion_ids[1::2] = 1
        cmd.time_steps[:] = 0
        cmd.start_bins[:] = cmd.motion_bin_starts[cmd.motion_ids]
        cmd._start_bin_valid[:] = True
        cmd._env.termination_manager.terminated[::2] = True
        boundary = cmd.bin_ranges[1][0]
        cmd.bin_start_failed_count[:boundary] = 0
        cmd.resample_time_steps(slice(None))
        self.assertEqual(cmd._current_start_count.sum().item(), 64)
        self.assertEqual(cmd._current_start_failed[:boundary].sum().item(), 32)
        self.assertEqual(cmd._current_start_failed[boundary:].sum().item(), 0)
        self.assertEqual(cmd._current_bin_failed[:boundary].sum().item(), 32)
        self.assertEqual(cmd._current_bin_failed[boundary:].sum().item(), 0)
        # Clip 0's curriculum has learned to remove help; clip 1 still needs it.
        self.assertEqual(cmd.assist_scale[cmd.motion_ids == 0].max().item(), 0)
        torch.testing.assert_close(
            cmd.assist_scale[cmd.motion_ids == 1],
            torch.full_like(cmd.assist_scale[cmd.motion_ids == 1], cmd.cfg.assist_beta_max),
        )

    def test_legacy_stance_settings_cannot_disable_tracking(self):
        for ranges in (None, [[(0, 1)], [(0, 1)]]):
            with self.subTest(ranges=ranges), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                cmd = self.command(stance_phase_ranges=[(0, 1)], motion_stance_phase_ranges=ranges)
                cmd.motion_ids[1::2] = 1
                for frame in (0, 59, 119):
                    cmd.time_steps[:] = frame
                    torch.testing.assert_close(cmd.standing_weight, torch.zeros(cmd.num_envs))
                    torch.testing.assert_close(cmd.tracking_weight, torch.ones(cmd.num_envs))
                    self.assertFalse(bool(cmd.is_stance.any()))

    def test_four_clips_sample_and_stop_at_their_own_boundaries(self):
        torch.manual_seed(42)
        cmd = self.command(motion_files=self.files, motion_stance_phase_ranges=None)
        cmd.resample_time_steps(slice(None))
        self.assertEqual(set(cmd.motion_ids.tolist()), {0, 1, 2, 3})
        torch.testing.assert_close(cmd.joint_pos[:, 0], cmd.motion_ids.float())
        self.assertTrue(bool((cmd.time_steps < cmd.motion_lengths).all()))
        # Exercise all four boundaries, including unequal transition-clip lengths.
        cmd.motion_ids[:] = torch.arange(cmd.num_envs) % 4
        cmd.time_steps[:] = cmd.motion_lengths - 2
        cmd._update_command()
        self.assertEqual(cmd.reference_frame_indices[:4].tolist(), [119, 359, 539, 839])
        self.assertTrue(bool(cmd.has_reached_end.all()))
        ids = cmd.motion_ids.clone()
        cmd._update_command()
        torch.testing.assert_close(cmd.motion_ids, ids)
        self.assertEqual(cmd.time_steps[:4].tolist(), [119, 239, 179, 299])

    def test_legacy_motion_file_config_still_works(self):
        cmd = self.command(motion_files=[], motion_file=self.files[0], motion_stance_phase_ranges=None)
        cmd.resample_time_steps(slice(None))
        self.assertEqual(cmd.num_motions, 1)
        self.assertTrue(bool((cmd.motion_ids == 0).all()))
        self.assertTrue(bool((cmd.time_steps < 120).all()))
        self.assertTrue(bool((cmd.joint_pos == 0).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
