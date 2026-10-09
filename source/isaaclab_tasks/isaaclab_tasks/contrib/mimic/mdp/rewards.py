# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import get_body_indices


def _phase_gate(env: ManagerBasedRLEnv, command_name: str, mode: str) -> torch.Tensor:
    """Deprecated compatibility gate: tracking is always active, standing is disabled."""
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    if mode == "tracking":
        return command.tracking_weight
    elif mode == "standing":
        return command.standing_weight
    raise ValueError(f"Unknown phase gate mode: {mode}")


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_tilt_error_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize anchor tilt error relative to the reference throughout the motion.

    Compares unit gravity vectors in the reference and robot anchor frames, ignoring
    world yaw. This penalty stays active throughout the reference and
    distinguishes large pre-takeoff tilt errors.
    Following a tilted reference incurs no penalty.

    Args:
        env: The environment.
        command_name: The motion command identifying the reference and robot anchor.

    Returns:
        Dimensionless squared gravity-vector distance in [0, 4], shape [N].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    gravity_w = torch.zeros_like(command.anchor_quat_w[:, :3])
    gravity_w[:, 2] = -1.0
    reference_gravity = quat_apply_inverse(command.anchor_quat_w, gravity_w)
    robot_gravity = quat_apply_inverse(command.robot_anchor_quat_w, gravity_w)
    return torch.sum(torch.square(robot_gravity - reference_gravity), dim=-1)


def motion_global_anchor_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward anchor linear velocity tracking in the fixed episode reference frame.

    Args:
        env: Environment containing the motion command.
        command_name: Name of the motion command term.
        std: Exponential kernel width [m/s].

    Returns:
        Exponential squared-error reward throughout the reference, shape [N].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    error = torch.sum(torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward anchor angular velocity tracking in the fixed episode reference frame.

    Args:
        env: Environment containing the motion command.
        command_name: Name of the motion command term.
        std: Exponential kernel width [rad/s].

    Returns:
        Exponential squared-error reward throughout the reference, shape [N].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    error = torch.sum(torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    body_indexes = get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    body_indexes = get_body_indices(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    body_indexes = get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    body_indexes = get_body_indices(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def standing_flat_orientation_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Same as `flat_orientation_l2`, but only penalized while the reference is in a stance interval."""
    return base_mdp.flat_orientation_l2(env, asset_cfg) * _phase_gate(env, command_name, "standing")


def standing_base_height_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Same as `base_height_l2`, but only penalized while the reference is in a stance interval."""
    return base_mdp.base_height_l2(env, target_height, asset_cfg, sensor_cfg) * _phase_gate(
        env, command_name, "standing"
    )


def standing_lin_vel_z_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Same as `lin_vel_z_l2`, but only penalized while the reference is in a stance interval."""
    return base_mdp.lin_vel_z_l2(env, asset_cfg) * _phase_gate(env, command_name, "standing")


def standing_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Same as `ang_vel_xy_l2`, but only penalized while the reference is in a stance interval."""
    return base_mdp.ang_vel_xy_l2(env, asset_cfg) * _phase_gate(env, command_name, "standing")


def standing_lin_vel_xy_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize planar base velocity while standing.

    This is the zero-command case of the velocity-tracking task reward the standing policy is trained
    with: during stance there is no trajectory to follow, the robot just has to hold still.
    """
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * _phase_gate(env, command_name, "standing")


def standing_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Same as `joint_deviation_l1`, but only penalized while the reference is in a stance interval."""
    return base_mdp.joint_deviation_l1(env, asset_cfg) * _phase_gate(env, command_name, "standing")


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]  # type: ignore
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
