# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.mimic import MOTION_TRACKING_DATA_DIR



# JOINT_NAMES = [
#     "left_hip_pitch_joint",
#     "right_hip_pitch_joint",
#     "waist_yaw_joint",
#     "left_hip_roll_joint",
#     "right_hip_roll_joint",
#     "waist_roll_joint",
#     "left_hip_yaw_joint",
#     "right_hip_yaw_joint",
#     "waist_pitch_joint",
#     "left_knee_joint",
#     "right_knee_joint",
#     "left_shoulder_pitch_joint",
#     "right_shoulder_pitch_joint",
#     "left_ankle_pitch_joint",
#     "right_ankle_pitch_joint",
#     "left_shoulder_roll_joint",
#     "right_shoulder_roll_joint",
#     "left_ankle_roll_joint",
#     "right_ankle_roll_joint",
#     "left_shoulder_yaw_joint",
#     "right_shoulder_yaw_joint",
#     "left_elbow_joint",
#     "right_elbow_joint",
#     "left_wrist_roll_joint",
#     "right_wrist_roll_joint",
#     "left_wrist_pitch_joint",
#     "right_wrist_pitch_joint",
#     "left_wrist_yaw_joint",
#     "right_wrist_yaw_joint",
# ]
# BODY_NAMES = [
#     "pelvis",
#     "left_hip_pitch_link",
#     "right_hip_pitch_link",
#     "waist_yaw_link",
#     "left_hip_roll_link",
#     "right_hip_roll_link",
#     "waist_roll_link",
#     "left_hip_yaw_link",
#     "right_hip_yaw_link",
#     "torso_link",
#     "left_knee_link",
#     "right_knee_link",
#     "left_shoulder_pitch_link",
#     "right_shoulder_pitch_link",
#     "left_ankle_pitch_link",
#     "right_ankle_pitch_link",
#     "left_shoulder_roll_link",
#     "right_shoulder_roll_link",
#     "left_ankle_roll_link",
#     "right_ankle_roll_link",
#     "left_shoulder_yaw_link",
#     "right_shoulder_yaw_link",
#     "left_elbow_link",
#     "right_elbow_link",
#     "left_wrist_roll_link",
#     "right_wrist_roll_link",
#     "left_wrist_pitch_link",
#     "right_wrist_pitch_link",
#     "left_wrist_yaw_link",
#     "right_wrist_yaw_link",
# ]

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# @configclass
# class G1CommandsCfg:
#     """Command specifications for the MDP."""

#     motion = mimic_mdp.MotionCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(1.0e9, 1.0e9),
#         debug_vis=False,
#         assist_eta=0.8,
#         assist_beta_max=0.8,
#         motion_quaternion_order="wxyz",
#         joint_names=JOINT_NAMES,
#         body_names=BODY_NAMES,
#         anchor_body_name="torso_link",
#         # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/g1_TO_jump_forward.npz"
#         # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_g1_retargeted.npz"
#         # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted_mirror_right.npz"
#         motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted_mirror_right_stance5s.npz",
#         # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted.npz"
#     )

@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    motion = mimic_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        assist_eta=0.8,
        assist_beta_max=0.8,
        # Select tracked joints/bodies in policy order; NPZ metadata supplies source ordering.
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        anchor_body_name="torso_link",
        motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/cmu_83_43.npz",
    )
