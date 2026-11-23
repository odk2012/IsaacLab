# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for placing each foot on its own contact position in the clip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import MotionContactCommand


def foot_contact_target_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot: int,
    std: float,
    asset_cfg: SceneEntityCfg,
    coarse_std: float | None = None,
) -> torch.Tensor:
    """``exp(-|p_foot - p_target|^2 / std^2)`` of one foot while it holds its target, else zero.

    ``foot`` is 0 (left) or 1 (right) and ``asset_cfg`` selects that robot foot's body. A foot holds its target
    (its last footstep, or its starting position) while the other foot steps, so each foot is paid only for
    staying on its own placed footstep; the stepping foot earns it once its footstep lands. Zero while the clip
    waits for the robot's feet, so stalling the clip does not pay.

    With ``coarse_std``, the reward is the mean of the ``std`` kernel and a wider ``coarse_std`` kernel: still 1 on
    the target, but a foot a few centimeters off is paid part of it, so placement improves from a gradient instead
    of from luck.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    error = _foot_error(command, foot, asset_cfg).square().sum(dim=-1)
    reward = torch.exp(-error / std**2)
    if coarse_std is not None:
        reward = 0.5 * (reward + torch.exp(-error / coarse_std**2))
    return reward * (command.in_contact[:, foot] & ~command.waiting).float()


def step_foot_target_exp(
    env: ManagerBasedRLEnv, command_name: str, foot: int, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``exp(-|p_foot - p_target|_xy^2 / std^2)`` of one foot while it steps toward its landing, else zero.

    The holding reward pays a foot only after its footstep lands; this pays the stepping foot every step for its
    horizontal distance to that landing, so the policy is drawn to step instead of standing on the holding foot.
    It is paid while the clip waits too, since the stepping foot is what the clip waits for.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    error = _foot_error(command, foot, asset_cfg)[:, :2].square().sum(dim=-1)
    return torch.exp(-error / std**2) * (~command.in_contact[:, foot]).float()


def _foot_error(command: MotionContactCommand, foot: int, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World offset [m] of a robot foot from its target. Shape is (num_envs, 3)."""
    position = command.robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    return position - command.contact_target_w[:, foot]


def step_foot_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot: int,
    asset_cfg: SceneEntityCfg,
    max_speed: float = 1.0,
    deadband: float = 0.02,
) -> torch.Tensor:
    """Horizontal speed [m/s] of one stepping foot toward its landing, clipped to ``[-max_speed, max_speed]``.

    Positive while the stepping foot moves toward its landing and negative while it moves away; zero for a holding
    foot and within ``deadband`` [m] of the landing, where the direction to the target is not meaningful.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    offset = -_foot_error(command, foot, asset_cfg)[:, :2]
    distance = offset.norm(dim=-1)
    velocity = command.robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids[0], :2]
    speed = (velocity * offset).sum(dim=-1) / distance.clamp(min=1e-6)
    active = ~command.in_contact[:, foot] & (distance > deadband)
    return speed.clamp(-max_speed, max_speed) * active.float()
