# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    joint_action_name: str = "JointPositionAction",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term(joint_action_name)._offset[env_ids, joint_ids] = pos  # type: ignore


# randomize_rigid_body_com is intentionally not redefined here -- isaaclab.envs.mdp already provides
# a backend-aware version (handles the PhysX full-pose vs. Newton position-only set_coms_index calling
# convention correctly), pulled in via the `from isaaclab.envs.mdp import *` in this package's
# __init__.py. A local redefinition here would shadow that and needs to be kept in sync by hand.


# Default body the assistive wrench acts about: the floating base. Override per-config with the
# ``base_body_name`` term param to drive any other body. The name must exist both on the robot and
# in the motion command's ``body_names``, since the reference for the same body is read from there.
_DEFAULT_ASSIST_BODY_NAME = "pelvis"

# Default PD gains (world frame). k_p_lin stays 0: the paper applies no base-position pull (it would
# fight the reference-state-initialized position). Override any of these with the matching term param.
_DEFAULT_KP_LIN = 0.0
_DEFAULT_KD_LIN = 10.0
_DEFAULT_KP_ANG = 120.0
_DEFAULT_KD_ANG = 5.0


def reset_root_state_from_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    pose_range: dict[str, tuple[float, float]] = {},
    velocity_range: dict[str, tuple[float, float]] = {},
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Place the robot's floating base on the motion reference's root, with optional randomization.

    ``pose_range`` and ``velocity_range`` are keyed by ``x``/``y``/``z``/``roll``/``pitch``/``yaw``;
    missing keys default to no offset. Position and velocity offsets are added, the orientation offset
    is composed onto the reference orientation.
    Initial XY and yaw define the fixed episode reference frame. Height, tilt,
    and velocity noise remain tracking errors.

    Pair this with :func:`reset_joint_state_from_reference`. Both call
    :meth:`MotionCommand.resample_time_steps`, which draws the episode's start frame if it has not been
    drawn yet, so the robot lands on the frame this episode actually begins at.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    command.resample_time_steps(env_ids)
    asset: Articulation = env.scene[asset_cfg.name]

    root_pos = command.root_pos_w[env_ids]
    root_ori = command.root_quat_w[env_ids]
    root_lin_vel = command.root_lin_vel_w[env_ids]
    root_ang_vel = command.root_ang_vel_w[env_ids]

    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in keys], device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    root_pos = root_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    root_ori = math_utils.quat_mul(orientations_delta, root_ori)
    command.set_reference_frame(env_ids, rand_samples[:, :2], rand_samples[:, 5])
    rotation = command.reference_quat_offset[env_ids]
    root_lin_vel = math_utils.quat_apply(rotation, root_lin_vel)
    root_ang_vel = math_utils.quat_apply(rotation, root_ang_vel)

    ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in keys], device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    root_lin_vel = root_lin_vel + rand_samples[:, :3]
    root_ang_vel = root_ang_vel + rand_samples[:, 3:]

    asset.write_root_state_to_sim(torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids)


def reset_joint_state_from_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    command_name: str,
    position_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the robot's joints to the motion reference's pose and velocity, with optional noise.

    ``position_range`` is a uniform offset added to every joint, then clipped to the soft joint limits.
    Joint velocities are taken from the reference unchanged.

    See :func:`reset_root_state_from_reference` for why this also calls
    :meth:`MotionCommand.resample_time_steps`.
    """
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    command.resample_time_steps(env_ids)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = command.joint_pos_robot_order[env_ids]
    joint_vel = command.joint_vel_robot_order[env_ids]

    joint_pos = joint_pos + sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    soft_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class AssistiveWrench(ManagerTermBase):
    r"""Model-based assistive wrench applied at the floating base (ZEST eqs 13a/13b).

    (https://arxiv.org/abs/2602.00401, "Assistive Wrench Automatic Curriculum" + supplementary S6):
    a virtual spatial wrench is applied at the robot's root body to bring it closer to the reference motion.
    The wrench is a PD controller on the base pose-tracking error plus a feedforward that compensates the
    nominal base dynamics, expressed in the world frame.
    The strength of the wrench is governed by the gain beta. With ``adaptive_scale=True`` this is the
    paper's automatic curriculum: a per-env gain frozen at reset from the measured failure rate of the
    clip bin the episode starts in (see :attr:`MotionCommand.bin_assist_scale`), so help concentrates on
    the parts of the motion the policy still fails and retires itself elsewhere. With
    ``adaptive_scale=False`` beta is the single scalar ``scale``, which a curriculum term can anneal on
    a fixed schedule (see ``mimic.mdp.curriculums.assistive_wrench_scale``). ``scale`` multiplies the
    adaptive gain too, so it can be left at 1.0.

    Assistance follows the existing adaptive gain throughout the reference;
    it is not gated by stance phases.

    Base targets use the command's fixed episode reference frame, including initial
    XY translation and yaw randomization.

    Must be registered as an ``interval`` event with ``interval_range_s=(0.0, 0.0)`` and
    ``is_global_time=True`` so it fires every env step on all envs.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        # Structural (fixes the inertia buffer below), so it is resolved once here rather than per step.
        self._base_body_name = cfg.params.get("base_body_name", _DEFAULT_ASSIST_BODY_NAME)
        self._base_body_id = self._asset.body_names.index(self._base_body_name)
        # The motion stores only the bodies listed in the command's ``body_names``, in that order, so
        # the reference for the base body lives at a different index than it does on the robot.
        self._motion_base_body_id: int | None = None

        self._total_mass = self._asset.data.default_mass.sum(dim=1, keepdim=True).to(self.device)  # (N, 1)
        self._base_inertia_b = (
            self._asset.data.default_inertia[:, self._base_body_id].to(self.device).reshape(self.num_envs, 3, 3)
        )  # (N, 3, 3)

        # Gravity acceleration vector in world frame, e.g. (0, 0, -9.81).
        self._gravity_w = torch.tensor(env.sim.cfg.gravity, device=self.device).unsqueeze(0)  # (1, 3)

        # Reference base accelerations are not stored in the motion data; they are finite-differenced
        # from the reference velocity buffers on first call (the motion command exists by then).
        self._ref_lin_acc_w: torch.Tensor | None = None
        self._ref_ang_acc_w: torch.Tensor | None = None

        # Optional force and torque arrows at the base.
        if cfg.params.get("visualize", False):
            from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

            arrow_base_scale = (0.4, 0.4, 1.0)
            force_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/assistive_wrench/force")  # type: ignore
            force_cfg.markers["arrow"].scale = arrow_base_scale
            torque_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/assistive_wrench/torque")  # type: ignore
            torque_cfg.markers["arrow"].scale = arrow_base_scale
            self._force_visualizer = VisualizationMarkers(force_cfg)
            self._torque_visualizer = VisualizationMarkers(torque_cfg)

    def _precompute_reference_accelerations(self, motion_command) -> None:
        """Forward-difference the reference base velocity buffers into world-frame accelerations.

        Computed per clip and indexed by the command's ``reference_frame_indices`` at
        runtime. Every clip's final frame is zeroed; clips are never differentiated together.
        """
        self._motion_base_body_id = motion_command.motion_body_names.index(self._base_body_name)
        base = self._motion_base_body_id
        self._ref_lin_acc_w, self._ref_ang_acc_w = motion_command.motion.body_accelerations(base, self._env.step_dt)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,  # always None when is_global_time=True
        command_term_name: str,
        scale: float = 0.0,
        adaptive_scale: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        base_body_name: str = _DEFAULT_ASSIST_BODY_NAME,
        k_p_lin: float = _DEFAULT_KP_LIN,
        k_d_lin: float = _DEFAULT_KD_LIN,
        k_p_ang: float = _DEFAULT_KP_ANG,
        k_d_ang: float = _DEFAULT_KD_ANG,
        visualize: bool = False,
        force_visual_gain: float = 0.002,
        torque_visual_gain: float = 0.005,
        arrow_origin_offset: float = 0.35,
    ):
        asset = self._asset
        motion_command = env.command_manager.get_term(command_term_name)  # type: ignore
        if self._ref_lin_acc_w is None:
            self._precompute_reference_accelerations(motion_command)

        base = self._base_body_id
        motion_base = self._motion_base_body_id
        frame_idx = motion_command.reference_frame_indices  # Packed rows, shape (N,).

        # Share the episode reference frame with tracking rewards and observations.
        rotation = motion_command.reference_quat_offset
        motion = motion_command.motion
        p_ref = (
            math_utils.quat_apply(rotation, motion._body_pos_w[frame_idx, motion_base])
            + motion_command.reference_pos_offset
            + env.scene.env_origins
        )
        q_ref = math_utils.quat_mul(rotation, motion._body_quat_w[frame_idx, motion_base])
        v_ref = math_utils.quat_apply(rotation, motion._body_lin_vel_w[frame_idx, motion_base])
        w_ref = math_utils.quat_apply(rotation, motion._body_ang_vel_w[frame_idx, motion_base])
        a_lin_ref = math_utils.quat_apply(rotation, self._ref_lin_acc_w[frame_idx])
        a_ang_ref = math_utils.quat_apply(rotation, self._ref_ang_acc_w[frame_idx])

        p = asset.data.body_link_pos_w[:, base]
        q = asset.data.body_link_quat_w[:, base]
        v = asset.data.body_link_lin_vel_w[:, base]
        w = asset.data.body_link_ang_vel_w[:, base]

        mass = self._total_mass  # (N, 1)
        rot = math_utils.matrix_from_quat(q)  # (N, 3, 3)
        inertia_w = rot @ self._base_inertia_b @ rot.transpose(-1, -2)  # (N, 3, 3)
        gravity = self._gravity_w  # (1, 3)
        inertia_apply = lambda x: torch.einsum("nij,nj->ni", inertia_w, x)  # noqa: E731

        # -- linear force (eq 13a): feedforward accel + velocity/position PD + gravity (weight) support --
        force = mass * (a_lin_ref + k_p_lin * (p_ref - p) + k_d_lin * (v_ref - v) - gravity)

        # -- angular torque (eq 13b): feedforward accel + orientation/rate PD + gyroscopic + gravity-torque comp --
        ori_err = math_utils.quat_box_minus(q_ref, q)  # (N, 3): hat{Phi} boxminus Phi (world axis-angle)
        # base -> whole-body CoM (world): moment arm for the gravity-torque comp, measured from the body the
        # torque acts about (``base`` = body 9), so it stays consistent with the body-9 PD error / inertia.
        # Preserve the original whole-robot CoM; root_com_pos_w describes only the root link.
        body_mass = asset.data.body_mass
        body_mass = body_mass.torch if hasattr(body_mass, "torch") else body_mass
        body_com_pos_w = asset.data.body_com_pos_w
        body_com_pos_w = body_com_pos_w.torch if hasattr(body_com_pos_w, "torch") else body_com_pos_w
        robot_com_w = (body_mass.unsqueeze(-1) * body_com_pos_w).sum(dim=1) / body_mass.sum(dim=1, keepdim=True)
        r_bcom = robot_com_w - p  # (N, 3)
        torque = (
            inertia_apply(a_ang_ref)
            + k_p_ang * inertia_apply(ori_err)
            + k_d_ang * inertia_apply(w_ref - w)
            + torch.cross(w, inertia_apply(w), dim=-1)
            - torch.cross(r_bcom, mass * gravity, dim=-1)
        )

        # -- resolve beta: per-env (adaptive, ZEST S6) or a single scalar (fixed curriculum) --
        if adaptive_scale:
            beta = scale * motion_command.assist_scale.unsqueeze(-1)  # (N, 1)
        else:
            beta = scale  # scalar

        # -- scale by beta: the applied wrench (world frame) for every env --
        applied_force = beta * force  # (N, 3)
        applied_torque = beta * torque  # (N, 3)

        # -- apply at the base body in the world frame --
        forces = applied_force.unsqueeze(1)  # (N, 1, 3)
        torques = applied_torque.unsqueeze(1)
        if env_ids is None:
            asset.instantaneous_wrench_composer.add_forces_and_torques(forces, torques, body_ids=[base], is_global=True)  # type: ignore
        else:
            asset.instantaneous_wrench_composer.add_forces_and_torques(
                forces[env_ids],
                torques[env_ids],
                env_ids=env_ids,
                body_ids=[base],
                is_global=True,  # type: ignore
            )

        # -- optional debug arrows: red = applied force, blue = applied torque, drawn at the base --
        if visualize and hasattr(self, "_force_visualizer"):
            self._draw_wrench_arrows(
                p, applied_force, applied_torque, force_visual_gain, torque_visual_gain, arrow_origin_offset
            )

    # ------------------------------------------------------------------
    # Debug visualization helpers
    # ------------------------------------------------------------------

    def _draw_wrench_arrows(
        self,
        base_pos_w: torch.Tensor,
        force_w: torch.Tensor,
        torque_w: torch.Tensor,
        force_gain: float,
        torque_gain: float,
        origin_offset: float,
    ) -> None:
        """Update the force (red) and torque (blue) arrow markers from the applied wrench.

        The arrow_x marker's tail sits at its translation, so drawing it straight at the base buries
        the inner half inside the pelvis link. Each arrow is therefore pushed out along its own
        pointing direction by ``origin_offset`` so its tail starts outside the link and is visible.
        """
        f_quat, f_scale, f_dir = self._vector_to_arrow(self._force_visualizer, force_w, force_gain)
        t_quat, t_scale, t_dir = self._vector_to_arrow(self._torque_visualizer, torque_w, torque_gain)
        f_pos = base_pos_w + origin_offset * f_dir
        t_pos = base_pos_w + origin_offset * t_dir
        self._force_visualizer.visualize(translations=f_pos, orientations=f_quat, scales=f_scale)
        self._torque_visualizer.visualize(translations=t_pos, orientations=t_quat, scales=t_scale)

    def _vector_to_arrow(
        self, visualizer: VisualizationMarkers, vec_w: torch.Tensor, gain: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map a per-env world-frame vector to an arrow_x marker (orientation, scale, unit direction).

        Follows the native velocity-command convention (``_resolve_xy_velocity_to_arrow``): start from
        the marker's default scale and stretch only its length (X) by ``gain * |vec|``, so the arrow's
        length encodes magnitude while its girth is unchanged. Envs with a near-zero vector (e.g. an
        env whose assist scale ``beta`` has annealed to 0) get zero scale so their arrows vanish.
        Orientation aims +X along the vector via yaw/pitch (roll = 0). The returned unit direction lets
        the caller offset the arrow's tail out of the link it is drawn on.
        """
        default_scale = visualizer.cfg.markers["arrow"].scale  # type: ignore # (sx, sy, sz); sx is the length axis
        magnitude = torch.linalg.norm(vec_w, dim=1)  # (N,)
        scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        scale[:, 0] *= gain * magnitude
        scale[magnitude < 1e-6] = 0.0  # hide arrows for envs with no (or fully annealed) assistance

        direction = torch.nn.functional.normalize(vec_w, dim=1)  # eps avoids NaN on zero vectors
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        pitch = torch.atan2(-direction[:, 2], torch.linalg.norm(direction[:, :2], dim=1))
        quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw)
        return quat, scale, direction
