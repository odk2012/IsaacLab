# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset onto the clip's pose at the episode's start frame, on the clip's terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import MotionContactCommand


def reset_to_motion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw the episode's clip and start frame, then write the clip's root and joint state there.

    ``joint_position_range`` [rad] adds uniform noise to the clip's joint positions, clipped to the soft limits.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    command.sample(env_ids)
    robot = command.robot
    state = command.reference_root_state(env_ids)
    robot.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=env_ids)
    joint_pos, joint_vel = command.reference_joint_state(env_ids)
    joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(*joint_position_range)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel.clone(), env_ids=env_ids)
