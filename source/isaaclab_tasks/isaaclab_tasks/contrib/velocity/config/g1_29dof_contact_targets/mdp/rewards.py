# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward for placing the feet on the clip's contact positions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import MotionContactCommand


def foot_contact_target_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Mean over the feet in contact in the clip of ``exp(-|p_foot - p_target|^2 / std^2)``.

    ``p_target`` is the clip's foot contact position (the ankle roll link, averaged over the contact) and
    ``asset_cfg`` selects the robot's left and right feet in that order. Zero while the clip has no foot in contact.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    feet = command.robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    error = (feet - command.contact_target_w[:, :2]).square().sum(dim=-1)
    stance = command.in_contact[:, :2].float()
    return (torch.exp(-error / std**2) * stance).sum(dim=-1) / stance.sum(dim=-1).clamp_min(1.0)
