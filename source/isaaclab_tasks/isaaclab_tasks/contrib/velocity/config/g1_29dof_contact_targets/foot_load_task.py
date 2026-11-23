"""Walking with optional per-foot world-Z load commands, in newtons."""
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, RewardTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.mdp import MirrorObservationTermCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_foot_scalars

from .agents.rsl_rl_ppo_cfg import G1ContactTargetWalkPPORunnerCfg
from .env_cfg import (
    ContactTargetObsCfg,
    G1ContactTargetCommandsCfg,
    G1ContactTargetObservationsCfg,
    G1ContactTargetRewardsCfg,
    G1ContactTargetWalkEnvCfg,
)
from .mdp.foot_load import measured_foot_fz, foot_fz_error


class FootLoadCommand(CommandTerm):
    """Automatic double-support loads, or an external target lasting until reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.gz = -float(env.cfg.sim.gravity[2])
        if self.gz <= 0:
            raise ValueError("Foot load commands require downward gravity")
        self.splits = torch.tensor([0.5, 0.6, 0.4], device=self.device)
        self.left_fraction = torch.full((self.num_envs,), 0.5, device=self.device)
        self.external = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.external_N = torch.full((self.num_envs, 2), float("nan"), device=self.device)

    @property
    def weight_N(self):
        return self.robot.data.body_mass.torch.sum(dim=1) * self.gz

    @property
    def target_N(self):
        motion = self._env.command_manager.get_term("motion")
        reference_double_support = motion.motion.reference_foot_contact[motion.frames].all(dim=1)
        fractions = torch.stack((self.left_fraction, 1.0 - self.left_fraction), dim=1)
        automatic = fractions * self.weight_N[:, None]
        automatic = automatic.masked_fill(~reference_double_support[:, None], float("nan"))
        return torch.where(self.external[:, None], self.external_N, automatic)

    @property
    def command(self):
        # Finite command for generic IsaacLab consumers. Validity is exposed separately.
        return torch.nan_to_num(self.target_N, nan=0.0)

    def set_target_N(self, values, env_ids=slice(None)):
        """Set [left, right] targets per selected env; NaN means unspecified."""
        dst = self.external_N[env_ids]
        value = torch.as_tensor(values, device=self.device, dtype=dst.dtype)
        if value.shape == (2,):
            value = value.expand_as(dst)
        if value.shape != dst.shape:
            raise ValueError(f"Expected {tuple(dst.shape)} or (2,), got {tuple(value.shape)}")
        if torch.isinf(value).any() or (value < 0).any():
            raise ValueError("Use nonnegative newtons or NaN")
        self.external_N[env_ids] = value
        self.external[env_ids] = True

    def use_automatic(self, env_ids=slice(None)):
        self.external[env_ids] = False

    def reset(self, env_ids=None):
        self.use_automatic(slice(None) if env_ids is None else env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        idx = torch.randint(len(self.splits), self.left_fraction[env_ids].shape,
                            device=self.device)
        self.left_fraction[env_ids] = self.splits[idx]

    def _update_command(self):
        pass  # Targets are derived from the current reference phase on access.

    def _update_metrics(self):
        pass


@configclass
class FootLoadCommandCfg(CommandTermCfg):
    class_type: type = FootLoadCommand
    resampling_time_range: tuple[float, float] = (1.0, 2.0)
    debug_vis: bool = False


def target_fraction(env):
    term = env.command_manager.get_term("foot_load")
    return term.command / term.weight_N[:, None].clamp_min(1e-6)


def target_validity(env):
    term = env.command_manager.get_term("foot_load")
    return torch.isfinite(term.target_N).float()


def normalized_load_mse(env):
    """Mean squared error over assigned feet, normalized by body weight squared."""
    term = env.command_manager.get_term("foot_load")
    measured, sample_valid = measured_foot_fz(env)
    error, valid = foot_fz_error(measured, term.target_N, sample_valid)
    normalized = torch.where(valid, error, 0.0) / term.weight_N[:, None].clamp_min(1e-6)
    return normalized.square().sum(dim=1) / valid.sum(dim=1).clamp_min(1)


@configclass
class ForceCommandsCfg(G1ContactTargetCommandsCfg):
    foot_load = FootLoadCommandCfg()


@configclass
class ForceCommandObsCfg(ContactTargetObsCfg):
    # Reuse Vasu's symmetry rule for per-foot scalars on both pairs.
    load_fraction = MirrorObservationTermCfg(func=target_fraction, mirror=mirror_foot_scalars)
    load_valid = MirrorObservationTermCfg(func=target_validity, mirror=mirror_foot_scalars)


@configclass
class ForceObservationsCfg(G1ContactTargetObservationsCfg):
    command: ForceCommandObsCfg = ForceCommandObsCfg()


@configclass
class ForceRewardsCfg(G1ContactTargetRewardsCfg):
    foot_load_error = RewardTermCfg(func=normalized_load_mse, weight=-1.0)


@configclass
class WalkForceEnvCfg(G1ContactTargetWalkEnvCfg):
    commands: ForceCommandsCfg = ForceCommandsCfg()
    observations: ForceObservationsCfg = ForceObservationsCfg()
    rewards: ForceRewardsCfg = ForceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Conservative default for the new 102-pad training task on the 4090.
        self.scene.num_envs = 1
        self.video_recorders = []


@configclass
class WalkForceRunnerCfg(G1ContactTargetWalkPPORunnerCfg):
    experiment_name = "g1_contact_targets_walk_foot_load"
    wandb_project = "g1_contact_targets_walk_foot_load"
