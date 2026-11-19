# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End of the clip, and the robot falling too far behind or off the clip's path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import MotionContactCommand


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The clip has played to its last frame."""
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    return command.has_reached_end


def root_far_from_clip(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Horizontal distance [m] between the robot's root and the clip's root above ``threshold``."""
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    error = command.robot.data.root_pos_w.torch[:, :2] - command.clip_root_pos_w[:, :2]
    return error.norm(dim=-1) > threshold
