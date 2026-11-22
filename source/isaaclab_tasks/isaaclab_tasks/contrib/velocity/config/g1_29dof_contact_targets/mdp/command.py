# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion clip played alongside each episode, giving the policy its upcoming foot and hand contacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from .motion import ContactMotionLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionContactCommand(CommandTerm):
    """A motion clip played in step with each episode; its contacts are the policy's targets.

    Each reset draws a clip and a start frame, moves the environment onto the clip's terrain column (column ``i``
    holds clip ``i``), and the reset event places the robot on the clip's pose there. The clip then advances one
    frame per control step. :attr:`command` is the clip's root velocity ``(v_x, v_y, w_z)`` in its root yaw frame,
    which the gait rewards use to tell standing from walking; the policy observes the contacts instead.
    """

    cfg: MotionContactCommandCfg

    def __init__(self, cfg: MotionContactCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.motion = ContactMotionLibrary(
            cfg.motion_files,
            self.robot.joint_names,
            self.robot.body_names[0],
            cfg.foot_body_names,
            env.step_dt,
            self.device,
            cfg.contact_margin,
            cfg.contact_speed,
            cfg.min_contact_frames,
        )
        self.num_motions = len(cfg.motion_files)
        terrain = env.scene.terrain
        if terrain.terrain_origins is None or terrain.terrain_origins.shape[1] != self.num_motions:
            raise ValueError("The terrain must have one column per motion clip (see motion_terrain_generator).")
        weights = cfg.motion_weights or [1.0] * self.num_motions
        self._weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._wait_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lifted = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._air_steps = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.waiting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the clip waits for the robot's feet this step. Shape is (num_envs,)."""
        for name in ("foot_target_error", "clip_progress", "root_error", "clip_wait"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"MotionContactCommand:\n\tClips: {self.num_motions}\n\tFrames: {int(self.motion.lengths.sum())}\n"

    """
    Clip state
    """

    @property
    def command(self) -> torch.Tensor:
        """Clip root velocity ``(v_x, v_y, w_z)`` [m/s, rad/s] in the clip root's yaw frame. Shape (num_envs, 3)."""
        return self._command

    @property
    def frames(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    @property
    def has_reached_end(self) -> torch.Tensor:
        return self.time_steps >= self.motion.lengths[self.motion_ids] - 1

    @property
    def clip_root_pos_w(self) -> torch.Tensor:
        return self.motion.root_pos[self.frames] + self._env.scene.env_origins

    @property
    def contact_target_w(self) -> torch.Tensor:
        """World position [m] of each contact body's contact target. Shape is (num_envs, 4, 3)."""
        return self.motion.contact_target[self.frames] + self._env.scene.env_origins[:, None, :]

    @property
    def in_contact(self) -> torch.Tensor:
        """Whether each contact body is in contact in the clip. Shape (num_envs, 4)."""
        return self.motion.in_contact[self.frames]

    """
    Episode sampling (from the reset event)
    """

    def sample(self, env_ids: torch.Tensor):
        """Draw the clip and start frame of the coming episode and move the environment onto the clip's terrain."""
        count = len(env_ids)
        if count == 0:
            return
        motion_ids = torch.multinomial(self._weights, count, replacement=True)
        length = self.motion.lengths[motion_ids]
        if self.cfg.random_start:
            # leave start_margin_s of the clip to play
            latest = (length - 1 - int(self.cfg.start_margin_s * self.motion.fps)).clamp_min(0)
            start = (torch.rand(count, device=self.device) * (latest + 1)).long()
            # some episodes still start from the clip's first pose
            first = torch.rand(count, device=self.device) < self.cfg.first_frame_probability
            start = torch.where(first, 0, start)
        else:
            start = torch.zeros_like(motion_ids)
        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = start
        self._wait_steps[env_ids] = 0
        self._lifted[env_ids] = False
        self._air_steps[env_ids] = 0
        self.waiting[env_ids] = False
        terrain = self._env.scene.terrain
        terrain.terrain_levels[env_ids] = 0
        terrain.terrain_types[env_ids] = motion_ids
        terrain.env_origins[env_ids] = terrain.terrain_origins[0, motion_ids]
        self._compute_command()

    def reference_root_state(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Clip root pose and velocity at each episode's frame in the world frame, shape (len(env_ids), 13)."""
        frames = self.motion.frame_indices(self.motion_ids[env_ids], self.time_steps[env_ids])
        return torch.cat(
            (
                self.motion.root_pos[frames] + self._env.scene.env_origins[env_ids],
                self.motion.root_quat[frames],
                self.motion.root_lin_vel[frames],
                self.motion.root_ang_vel[frames],
            ),
            dim=-1,
        )

    def reference_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip joint positions [rad] and velocities [rad/s] at each episode's frame, in articulation order."""
        frames = self.motion.frame_indices(self.motion_ids[env_ids], self.time_steps[env_ids])
        return self.motion.joint_pos[frames], self.motion.joint_vel[frames]

    """
    Implementation specific functions.
    """

    def _compute_command(self):
        frames = self.frames
        yaw = math_utils.yaw_quat(self.motion.root_quat[frames])
        self._command[:, :2] = math_utils.quat_apply_inverse(yaw, self.motion.root_lin_vel[frames])[:, :2]
        self._command[:, 2] = self.motion.root_ang_vel[frames, 2]

    def _update_metrics(self):
        feet = self.robot.data.body_pos_w.torch[:, self._foot_ids]
        error = (feet - self.contact_target_w[:, :2]).norm(dim=-1)
        stance = self.in_contact[:, :2].float()
        self.metrics["foot_target_error"][:] = (error * stance).sum(-1) / stance.sum(-1).clamp_min(1.0)
        self.metrics["clip_progress"][:] = self.time_steps / (self.motion.lengths[self.motion_ids] - 1).clamp_min(1)
        self.metrics["root_error"][:] = (self.robot.data.root_pos_w.torch[:, :2] - self.clip_root_pos_w[:, :2]).norm(
            dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # episodes are drawn by the reset event through sample()
        pass

    def _update_command(self):
        last = self.motion.lengths[self.motion_ids] - 1
        following = torch.minimum(self.time_steps + 1, last)
        if self.cfg.wait_for_contact:
            following = self._paced_time_steps(following)
        self.time_steps = following
        self._compute_command()

    def _paced_time_steps(self, following: torch.Tensor) -> torch.Tensor:
        """Advance the clip at the robot's pace (see :attr:`MotionContactCommandCfg.wait_for_contact`)."""
        motion = self.motion
        starts = motion.starts[self.motion_ids]
        origins = self._env.scene.env_origins[:, None, :]
        feet = self.robot.data.body_pos_w.torch[:, self._foot_ids]
        sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        if not hasattr(self, "_sensor_foot_ids"):
            self._sensor_foot_ids = sensor.find_bodies(self.cfg.foot_body_names, preserve_order=True)[0]
        down = sensor.data.net_forces_w.torch[:, self._sensor_foot_ids].norm(dim=-1) > self.cfg.contact_force_threshold

        def reached(frames: torch.Tensor) -> torch.Tensor:
            target = motion.contact_target[frames, :2] + origins
            return down & ((feet - target).norm(dim=-1) < self.cfg.reach_radius)

        # wait: the clip does not plant a foot until the robot's foot is down on that contact
        touchdown = motion.touchdown[starts + following, :2]
        wait = (touchdown & ~reached(starts + following)).any(dim=-1)
        if self.cfg.max_wait_s is not None:
            wait &= self._wait_steps < round(self.cfg.max_wait_s / self._env.step_dt)
        self._wait_steps = torch.where(wait, self._wait_steps + 1, 0)
        self.waiting[:] = wait
        paced = torch.where(wait, self.time_steps, following)
        # move up: a swinging foot that is down on its next contact jumps the clip to that touchdown
        frames = self.frames
        upcoming = motion.next_touchdown[frames, :2]
        upcoming_frames = starts[:, None] + upcoming.clamp_min(0)
        targeting = (
            motion.contact_target[frames, :2]
            - motion.contact_target[upcoming_frames, torch.arange(2, device=self.device)]
        ).norm(dim=-1) < 1.0e-6
        # only a new touchdown counts: since the clip foot lifted off, the robot foot has been off the ground for
        # min_air_s (so the drop after a reset and bounces do not count)
        swinging = ~motion.in_contact[frames, :2]
        self._air_steps = torch.where(down, 0, self._air_steps + 1)
        lifted_now = self._air_steps >= round(self.cfg.min_air_s / self._env.step_dt)
        self._lifted = swinging & (self._lifted | lifted_now)
        early = (upcoming >= 0) & swinging & targeting & self._lifted & reached(frames)
        jump = torch.where(early, upcoming, torch.full_like(upcoming, torch.iinfo(torch.long).max)).amin(dim=-1)
        self.metrics["clip_wait"][:] = wait.float()
        return torch.where(early.any(dim=-1), torch.maximum(jump, paced), paced)

    @property
    def _foot_ids(self) -> list[int]:
        if not hasattr(self, "_foot_body_ids"):
            self._foot_body_ids = [self.robot.body_names.index(name) for name in self.cfg.foot_body_names]
        return self._foot_body_ids

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "target_visualizer"):
            self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
        if hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            # markers 0 (in contact) and 1 (upcoming) per contact body
            targets = self.contact_target_w.reshape(-1, 3)
            indices = (~self.in_contact).long().reshape(-1)
            self.target_visualizer.visualize(translations=targets, marker_indices=indices)


@configclass
class MotionContactCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionContactCommand`."""

    class_type: type = MotionContactCommand

    asset_name: str = "robot"

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    """Unused: clips are drawn at reset."""

    motion_files: list[str] = MISSING
    """Clips; terrain column ``i`` must be built from clip ``i``."""

    motion_weights: list[float] | None = None
    """Relative chance of drawing each clip; None draws them uniformly."""

    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    """Robot (and clip) bodies whose positions are the foot contact targets, left first."""

    random_start: bool = True
    """Start episodes at a random clip frame (see :attr:`first_frame_probability`); otherwise at the first frame."""

    first_frame_probability: float = 0.3
    """With :attr:`random_start`, the chance that an episode still starts at the clip's first frame."""

    start_margin_s: float = 1.0
    """Clip time [s] left to play at the latest random start."""

    contact_margin: float = 0.01
    """Distance [m] from the terrain within which two clip foot box corners, or a hand sphere, touch it."""

    contact_speed: float = 0.3
    """Speed [m/s] below which a touching clip ankle or hand counts as in contact (excludes dragged feet)."""

    min_contact_frames: int = 3
    """Shortest contact and gap [frames] kept when measuring the clip's contacts."""

    wait_for_contact: bool = False
    """Advance the clip at the robot's pace instead of one frame per step.

    The clip does not plant a foot (so the other foot gets no new target) until the robot's foot is down within
    :attr:`reach_radius` of that contact; and when the robot lifts a foot the clip swings (for at least
    :attr:`min_air_s`) and puts it down within :attr:`reach_radius` of its next contact early, the clip jumps to that
    touchdown.
    """

    max_wait_s: float | None = None
    """Longest time [s] the clip waits for one touchdown before it plays on regardless; None waits indefinitely."""

    min_air_s: float = 0.1
    """Shortest time [s] a robot foot must be off the ground for its next touchdown to move the clip up."""

    reach_radius: float = 0.15
    """Distance [m] from a contact target within which a robot foot that is down has reached it."""

    contact_sensor_name: str = "contact_forces"
    """Contact sensor telling whether the robot's feet are down (used with :attr:`wait_for_contact`)."""

    contact_force_threshold: float = 5.0
    """Net foot force [N] above which a robot foot is down."""

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/contact_targets",
        markers={
            "in_contact": sim_utils.SphereCfg(
                radius=0.04, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1))
            ),
            "upcoming": sim_utils.SphereCfg(
                radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9))
            ),
        },
    )
    """Markers at the contact targets: green for bodies in contact, blue for upcoming contacts."""
