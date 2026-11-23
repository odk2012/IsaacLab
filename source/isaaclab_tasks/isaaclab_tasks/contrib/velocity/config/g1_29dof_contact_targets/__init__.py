# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-terrain G1 following the foot and hand contacts of stair and walking clips (or walking clips only)."""

import gymnasium as gym

# stair and walking clips, and the walking clips only
for task, cfg_name, runner in (("", "", ""), ("-Walk", "Walk", "Walk")):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets{task}{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:G1ContactTarget{cfg_name}EnvCfg{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ContactTarget{runner}PPORunnerCfg",
            },
        )

gym.register(
    id="IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets-Walk-Force",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.foot_load_task:WalkForceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.foot_load_task:WalkForceRunnerCfg",
    },
)
