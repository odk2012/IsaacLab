# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-terrain G1 that follows motion clips' foot and hand contacts, without perception.

Each episode plays a stair or walking clip on that clip's terrain. The policy observes, besides proprioception,
each foot's and hand's current or next contact in the clip; the rewards are the rigid task's style, gait, and
regularization terms (no velocity tracking) and rewards for stepping toward and onto the clip's footsteps.
"""

from pathlib import Path

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.env_cfg.motion_cfg import (
    STAIR_MOTION_FILES,
)

from ..g1_29dof_rigid.env_cfg import G1EventCfg, G1RewardsRoughCfg, G1TerminationsCfg
from ..g1_29dof_rigid.env_cfg.observation_cfg import CriticHistoryCfg, PolicyHistoryCfg, PrivilegedHistoryCfg
from ..g1_29dof_rigid.rough_env_cfg import G1RoughEnvCfg
from . import mdp

WALK_MOTION_DIR = Path(__file__).resolve().parent / "data" / "motion_library"

WALK_MOTION_FILES = [str(WALK_MOTION_DIR / f"kimodo_ramp_seed{seed}.npz") for seed in (45, 46, 47)] + [
    str(WALK_MOTION_DIR / f"kimodo_platform_seed{seed}.npz") for seed in (44, 45)
]
"""Walking clips from motion_library: up a 0.1 m ramp, and onto a 0.15 m platform."""

TEST_MOTION_FILES = [STAIR_MOTION_FILES[3], WALK_MOTION_FILES[2]]
"""Held-out clips, never drawn in training and the only ones in ``-Play``: the pyramid climbed from its fourth side
(``motion_yaw_270``) and the third ramp walk (``kimodo_ramp_seed47``)."""

TRAIN_STAIR_FILES = [path for path in STAIR_MOTION_FILES if path not in TEST_MOTION_FILES]
TRAIN_WALK_FILES = [path for path in WALK_MOTION_FILES if path not in TEST_MOTION_FILES]
TEST_WALK_FILES = [path for path in WALK_MOTION_FILES if path in TEST_MOTION_FILES]

MOTION_FILES = TRAIN_STAIR_FILES + TRAIN_WALK_FILES
"""Training clips: three pyramid stair clips (approached from three sides), then four walking clips; terrain column
``i`` holds clip ``i``."""

MOTION_WEIGHTS = [1.0 / len(TRAIN_STAIR_FILES)] * len(TRAIN_STAIR_FILES) + [1.0 / len(TRAIN_WALK_FILES)] * len(
    TRAIN_WALK_FILES
)
"""Relative chance of each training clip: stairs and walking in equal shares."""


@configclass
class G1ContactTargetCommandsCfg:
    """The clip played with each episode; no velocity command."""

    motion = mdp.MotionContactCommandCfg(
        motion_files=MOTION_FILES,
        motion_weights=MOTION_WEIGHTS,
        debug_vis=True,
        # the clip waits (up to 1 s per touchdown) for the robot's feet to reach their contacts
        wait_for_contact=True,
        max_wait_s=1.0,
        # every episode starts from the clip's first pose; the foot targets then follow the clip one footstep at a time
        random_start=False,
        # a foot has reached its target when it is down within 8 cm of it
        reach_radius=0.08,
    )


@configclass
class ContactTargetObsCfg(ObsGroup):
    """The clip's current or next foot and hand contacts, relative to the robot (actor and critic)."""

    upcoming_contacts = ObsTerm(
        func=mdp.upcoming_contacts_b, mirror=mdp.mirror_upcoming_contacts, params={"command_name": "motion"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ContactTargetObservationsCfg:
    """The rigid task's proprioception (no height scan) with the clip's contacts in place of the velocity command."""

    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    command: ContactTargetObsCfg = ContactTargetObsCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()


@configclass
class G1ContactTargetRewardsCfg(G1RewardsRoughCfg):
    """The rigid task's rough-terrain rewards without velocity tracking, and per-foot contact target rewards.

    The gait terms that tell standing from walking read the clip's root velocity (the ``motion`` command). Each foot
    is paid for holding its target (1 on target, 0.88 at 5 cm, 0.63 at 10 cm) and, while it steps, for being near
    its landing and for moving toward it.
    """

    track_lin_vel_xy = None
    track_ang_vel_z = None
    track_heading = None
    foot_clearance = None

    left_foot_contact_target = RewTerm(
        func=mdp.foot_contact_target_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "foot": 0,
            "std": 0.1,
            "coarse_std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
        },
    )
    left_foot_step_target = RewTerm(
        func=mdp.step_foot_target_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "foot": 0,
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
        },
    )
    left_foot_progress = RewTerm(
        func=mdp.step_foot_progress,
        weight=1.0,
        params={
            "command_name": "motion",
            "foot": 0,
            "asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
        },
    )
    right_foot_contact_target = RewTerm(
        func=mdp.foot_contact_target_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "foot": 1,
            "std": 0.1,
            "coarse_std": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
        },
    )
    right_foot_step_target = RewTerm(
        func=mdp.step_foot_target_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "foot": 1,
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
        },
    )
    right_foot_progress = RewTerm(
        func=mdp.step_foot_progress,
        weight=1.0,
        params={
            "command_name": "motion",
            "foot": 1,
            "asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
        },
    )

    def __post_init__(self):
        for term in vars(self).values():
            if isinstance(term, RewTerm) and term.params.get("command_name") == "base_velocity":
                term.params["command_name"] = "motion"


@configclass
class G1ContactTargetEventCfg(G1EventCfg):
    """The rigid task's randomization; every episode starts on its clip's pose."""

    reset_base = EventTerm(
        func=mdp.reset_to_motion,
        mode="reset",
        params={"command_name": "motion", "joint_position_range": (-0.05, 0.05)},
    )


@configclass
class G1ContactTargetTerminationsCfg(G1TerminationsCfg):
    """Falls as in the rigid task; the clip's end; and the robot 1 m away from the clip's root."""

    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    root_far_from_clip = DoneTerm(func=mdp.root_far_from_clip, params={"command_name": "motion", "threshold": 1.0})


@configclass
class G1ContactTargetCurriculumCfg:
    """No curriculum: the rigid task's terms all ramp velocity commands or tracking rewards."""


@configclass
class G1ContactTargetEnvCfg(G1RoughEnvCfg):
    """Follow the foot and hand contacts of stair and walking clips on their own rigid terrain."""

    commands: G1ContactTargetCommandsCfg = G1ContactTargetCommandsCfg()
    observations: G1ContactTargetObservationsCfg = G1ContactTargetObservationsCfg()
    rewards: G1ContactTargetRewardsCfg = G1ContactTargetRewardsCfg()
    events: G1ContactTargetEventCfg = G1ContactTargetEventCfg()
    terminations: G1ContactTargetTerminationsCfg = G1ContactTargetTerminationsCfg()
    curriculum: G1ContactTargetCurriculumCfg = G1ContactTargetCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # the clip reset writes the root and joints; the rigid task's reset parameters do not apply
        self.events.reset_base.params = {"command_name": "motion", "joint_position_range": (-0.05, 0.05)}
        self.events.reset_robot_joints = None
        # one terrain column per clip; the command moves each robot onto its clip's column at reset
        self.scene.terrain.terrain_generator = mdp.motion_terrain_generator(self.commands.motion.motion_files)
        self.scene.terrain.max_init_terrain_level = None
        # the clips last 4 to 7.5 s, longer with the waits; the clip's end usually ends the episode first
        self.episode_length_s = 20.0

    def play_mode(self):
        super().play_mode()
        # the velocity play mode shrinks the terrain to a random 5x5 grid; keep one column per clip
        self.scene.terrain.terrain_generator = mdp.motion_terrain_generator(self.commands.motion.motion_files)


@configclass
class G1ContactTargetEnvCfg_PLAY(G1ContactTargetEnvCfg):
    """The held-out clips (:data:`TEST_MOTION_FILES`) from their first frame, without pushes or observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 18
        _configure_play(self, TEST_MOTION_FILES)


def _configure_play(cfg: G1ContactTargetEnvCfg, motion_files: list[str]):
    """Play ``motion_files`` from their first frame at the robot's pace, without pushes or observation noise."""
    cfg.scene.num_envs = 18
    cfg.commands.motion.motion_files = motion_files
    cfg.commands.motion.motion_weights = None
    cfg.scene.terrain.terrain_generator = mdp.motion_terrain_generator(motion_files)
    # the clip waits as long as it takes for the robot's feet to reach their contacts
    cfg.commands.motion.max_wait_s = None
    cfg.episode_length_s = 30.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    # the camera looks at env 0's clip terrain (it does not follow the robot); the sidebar's "Live Plots" section
    # plots env 0's reward terms every step
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            origin_type="env",
            origin_env_index=0,
            eye=(-4.0, -4.0, 3.0),
            lookat=(0.0, 0.0, 0.5),
            enable_live_plots=True,
            live_plots_update_interval=1,
        )
    ]
    cfg.video_recorders = []


@configclass
class G1ContactTargetWalkEnvCfg(G1ContactTargetEnvCfg):
    """The same task on the walking clips only (:data:`TRAIN_WALK_FILES`, drawn uniformly), without stairs."""

    def __post_init__(self):
        self.commands.motion.motion_files = TRAIN_WALK_FILES
        self.commands.motion.motion_weights = None
        super().__post_init__()


@configclass
class G1ContactTargetWalkEnvCfg_PLAY(G1ContactTargetWalkEnvCfg):
    """The held-out walking clip (:data:`TEST_WALK_FILES`) from its first frame."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play(self, TEST_WALK_FILES)
