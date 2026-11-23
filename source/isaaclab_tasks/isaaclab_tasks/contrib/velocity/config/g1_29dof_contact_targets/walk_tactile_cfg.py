"""Sole tactile configuration for the G1 contact-target walking task."""
from copy import deepcopy
from pathlib import Path

from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.hydra import preset

from ..g1_29dof_rigid.env_cfg import G1SceneCfg


@configclass
class WalkTactileSceneCfg(G1SceneCfg):
    tactile_pad_gpu: SensorBaseCfg | None = None


def configure_walk_tactile(cfg):
    from isaaclab.sensors.tactile_array.tactile_pad_gpu import TactilePadGPUCfg

    physics = cfg.sim.physics
    native = physics.newton_mjwarp
    pad_physics = deepcopy(native)
    pad_physics.solver_cfg.nconmax = 4096
    pad_physics.solver_cfg.njmax = 32768
    other = ("physx", "isaacsim_physx", "ovphysx", "newton_kamino")
    cfg.sim.physics = preset(
        default=pad_physics, newton_mjwarp=pad_physics,
        **{name: getattr(physics, name) for name in other},
    )

    original = cfg.scene.robot.spawn.usd_path
    pad_asset = str(Path(original).with_name("g1_29dof_tactile_pad102.usda"))
    cfg.scene.robot.spawn.usd_path = preset(
        default=pad_asset, newton_mjwarp=pad_asset,
        **dict.fromkeys(other, original),
    )
    pad = TactilePadGPUCfg(
        prim_path="{ENV_REGEX_NS}/Robot", update_period=0.0, force_threshold=1.0
    )
    cfg.scene.tactile_pad_gpu = preset(
        default=pad, newton_mjwarp=pad, **dict.fromkeys(other),
    )
