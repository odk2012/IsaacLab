# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg


@configclass
class G1ContactTargetPPORunnerCfg(G1RoughPPORunnerCfg):
    """The rigid task's PPO runner; the actor reads proprioception and the clip's contacts (``command``)."""

    wandb_project = "g1_29dof_rigid_contact_targets_ppo"
    experiment_name = "g1_29dof_rigid_contact_targets_ppo"


@configclass
class G1ContactTargetWalkPPORunnerCfg(G1ContactTargetPPORunnerCfg):
    """The same runner for the walking-only task, logged to its own project."""

    wandb_project = "g1_29dof_rigid_contact_targets_walk_ppo"
    experiment_name = "g1_29dof_rigid_contact_targets_walk_ppo"
