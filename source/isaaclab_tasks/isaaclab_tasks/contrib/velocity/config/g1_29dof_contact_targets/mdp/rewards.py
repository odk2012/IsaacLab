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
    env: ManagerBasedRLEnv, command_name: str, foot: int, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``exp(-|p_foot - p_target|^2 / std^2)`` of one foot while the clip has that foot in contact, else zero.

    ``foot`` is 0 (left) or 1 (right) and ``asset_cfg`` selects that robot foot's body. ``p_target`` is that foot's
    contact position in the clip (its ankle roll link, averaged over the contact), so each foot is scored only
    against its own contact.
    """
    command: MotionContactCommand = env.command_manager.get_term(command_name)
    position = command.robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
    error = (position - command.contact_target_w[:, foot]).square().sum(dim=-1)
    return torch.exp(-error / std**2) * command.in_contact[:, foot].float()
