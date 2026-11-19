# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The clip's upcoming foot and hand contacts, relative to the robot, and their left/right mirror."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import MotionContactCommand


def upcoming_contacts_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """For each contact body (left foot, right foot, left hand, right hand): the current or next contact.

    Per body, 5 values: the contact position [m] relative to the root in the root yaw frame, whether the body is in
    contact in the clip (0 or 1), and the time until the contact begins as a fraction of the horizon (0 in contact,
    1 when no contact is announced). Shape is (num_envs, 20).
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    yaw = math_utils.yaw_quat(robot.data.root_quat_w.torch)[:, None, :].expand(-1, 4, -1)
    offset = command.contact_target_w - robot.data.root_pos_w.torch[:, None, :]
    position = math_utils.quat_apply_inverse(yaw, offset)
    contact = command.in_contact.float().unsqueeze(-1)
    time = (command.time_to_contact / command.motion.horizon).unsqueeze(-1)
    return torch.cat((position, contact, time), dim=-1).reshape(env.num_envs, -1)


def mirror_upcoming_contacts(data: torch.Tensor) -> torch.Tensor:
    """Reflect :func:`upcoming_contacts_b` across the root XZ plane: swap left and right, negate local Y."""
    data = data.reshape(*data.shape[:-1], 4, 5)[..., [1, 0, 3, 2], :].clone()
    data[..., 1] = -data[..., 1]
    return data.reshape(*data.shape[:-2], 20)
