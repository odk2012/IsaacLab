# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from .env_cfg import (
    G1ActionsCfg,
    G1CommandsCfg,
    G1CurriculumCfg,
    G1EventCfg,
    G1ObservationsCfg,
    G1PhysicsCfg,
    G1RewardsRoughCfg,
    G1SceneCfg,
    G1TerminationsCfg,
)


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # physics settings owned by the task instead of inherited from the velocity base env
    sim: SimulationCfg = SimulationCfg(physics=G1PhysicsCfg())  # type: ignore
    rewards: G1RewardsRoughCfg = G1RewardsRoughCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    events: G1EventCfg = G1EventCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    seed: int = 42

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Randomization
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # self.sim.visualizer_cfgs = [
        #     NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=False),
        #     # NewtonRTXVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
        #     # KitVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
        # ]

        self.sim.visualizer_cfgs = []

        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir="logs/videos",
                video_length=200,
                video_interval=2000
            ),
            # VideoRecorderCfg(source="visualizer:kit", output_dir="videos/"),
        ]


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.command_vel = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.base_external_force_torque = None
        self.events.push_robot = None  # type: ignore

        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            NewtonRTXVisualizerCfg(eye=(0.0, -4.0, 1.0)),
            # KitVisualizerCfg(eye=(0.0, -4.0, 1.0)),
        ]
        self.video_recorders = [
            # VideoRecorderCfg(
            #     source="visualizer:newton_rtx",
            #     output_dir="logs/videos",
            #     video_length=200,
            #     video_interval=2000
            #     ),
        ]
