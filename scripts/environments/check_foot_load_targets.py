"""Check signed foot-load errors and log target/measured Fz in the tactile Walk task."""
import argparse
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab.app import launch_simulation
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab_rl.entrypoints.simple_agents import create_zero_action_policy
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.contrib.velocity.config.g1_29dof_contact_targets.mdp.foot_load import (
    foot_fz_error, measured_foot_fz,
)

TASK = "IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets-Walk"
SPLITS = {"50_50": (0.5, 0.5), "60_40": (0.6, 0.4), "40_60": (0.4, 0.6)}


def check_sign_and_mask():
    measured = torch.tensor([[0., 25.], [60., 40.], [10., 5.], [0., 0.]])
    target = torch.tensor([[150., 0.], [50., 50.], [float("nan"), 5.], [20., 0.]])
    error, valid = foot_fz_error(measured, target, torch.tensor([True, True, True, False]))
    expected = torch.tensor([[-150., 25.], [10., -10.],
                             [float("nan"), 0.], [float("nan"), float("nan")]])
    torch.testing.assert_close(error, expected, equal_nan=True)
    assert torch.equal(valid, torch.isfinite(expected))
    print("SIGN_MASK_OK: missed contact, zero target, missing target, reset, left/right")


def make_target(weight, phase):
    if phase == "unspecified":
        return weight.new_full((weight.shape[0], 2), float("nan"))
    return weight[:, None] * weight.new_tensor(SPLITS[phase])


def record(env, out):
    raw = env.unwrapped
    robot = raw.scene["robot"]
    pad = raw.scene.sensors["tactile_pad_gpu"]
    gravity = tuple(float(v) for v in raw.cfg.sim.gravity)
    if abs(gravity[0]) + abs(gravity[1]) > 1e-6 or gravity[2] >= 0:
        raise ValueError(f"Expected vertical downward gravity, got {gravity}")

    env.reset()
    zero_action = create_zero_action_policy(env)
    sample = None
    valid_steps = resets = double_support_steps = 0
    phases = {k: [[], []] for k in SPLITS}
    fields = ["step", "time_s", "phase", "reset", "sensor_valid",
              "double_support", "weight_N"]
    for side in ("left", "right"):
        fields += [f"{side}_target_N", f"{side}_measured_N",
                   f"{side}_error_N", f"{side}_error_valid"]

    with (out / "target-measured-fz.csv").open("w", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, 136):
            phase = ("unspecified" if step <= 15 else
                     "50_50" if step <= 55 else
                     "60_40" if step <= 95 else "40_60")

            # Set the target before the action. A reset at the end invalidates this pair.
            weight = robot.data.body_mass.torch.sum(dim=1) * (-gravity[2])
            if not (torch.isfinite(weight).all() and (weight > 0).all()):
                raise RuntimeError(f"Invalid simulated body weight: {weight}")
            target = make_target(weight, phase)
            _, _, terminated, truncated, _ = env.step(zero_action())
            done = (terminated | truncated).reshape(-1)

            measured, sensor_valid = measured_foot_fz(raw)
            if pad.health.any().item():
                raise RuntimeError(f"Tactile sensor health flags: {pad.health.tolist()}")
            usable = sensor_valid & ~done
            if usable[0].item() and not torch.isfinite(measured[0]).all().item():
                raise RuntimeError(f"Non-finite foot force at valid step {step}")
            error, mask = foot_fz_error(measured, target, usable)

            reset = bool(done[0].item())
            good = bool(usable[0].item())
            total_weight = float(weight[0].item())
            actual = measured[0].tolist()
            desired = target[0].tolist()
            delta = error[0].tolist()
            scored = mask[0].tolist()
            both = good and min(actual) > 5.0
            valid_steps += int(good)
            resets += int(reset)
            double_support_steps += int(both)

            # Pick the real double-support sample whose total Fz is closest to body weight.
            # This is a fixed input for comparing target splits, not a standing-success test.
            if both:
                mismatch = abs(sum(actual) - total_weight)
                if sample is None or mismatch < sample[0]:
                    sample = (mismatch, step, total_weight, actual.copy())

            row = dict(step=step, time_s=step * float(raw.step_dt), phase=phase,
                       reset=int(reset), sensor_valid=int(sensor_valid[0].item()),
                       double_support=int(both), weight_N=total_weight)
            for i, side in enumerate(("left", "right")):
                row[f"{side}_target_N"] = desired[i] if math.isfinite(desired[i]) else ""
                row[f"{side}_measured_N"] = actual[i] if good else ""
                row[f"{side}_error_N"] = delta[i] if scored[i] else ""
                row[f"{side}_error_valid"] = int(scored[i])
                if scored[i]:
                    phases[phase][i].append(delta[i])
            writer.writerow(row)

    if not valid_steps or any(not all(per_foot) for per_foot in phases.values()):
        raise RuntimeError("One or more target phases had no valid paired readings; inspect CSV.")

    summary = dict(control_steps=135, valid_sensor_steps=valid_steps, reset_steps_count=resets,
                   double_support_steps=double_support_steps, fixed_sample=None)
    summary["phase_counts_left_right"] = {
        k: [len(values) for values in per_foot] for k, per_foot in phases.items()
    }
    if sample is not None:
        _, step, weight_N, actual = sample
        force = torch.tensor([actual], dtype=torch.float64)
        weight = torch.tensor([weight_N], dtype=torch.float64)
        comparisons = {}
        for phase in SPLITS:
            target = make_target(weight, phase)
            err, mask = foot_fz_error(force, target, torch.tensor([True]))
            assert mask.all()
            torch.testing.assert_close(target.sum(dim=1), weight)
            comparisons[phase] = dict(target_N=target[0].tolist(), error_N=err[0].tolist())
        e50, e60, e40 = [torch.tensor(comparisons[k]["error_N"], dtype=torch.float64)
                         for k in ("50_50", "60_40", "40_60")]
        shift = weight_N * torch.tensor([-0.1, 0.1], dtype=torch.float64)
        torch.testing.assert_close(e60 - e50, shift)
        torch.testing.assert_close(e40 - e50, -shift)
        summary["fixed_sample"] = dict(step=step, weight_N=weight_N,
                                       measured_N=actual, comparisons=comparisons)
        print("DOUBLE_SUPPORT_SPLITS_OK:", json.dumps(summary["fixed_sample"]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    check_sign_and_mask()
    cfg, _ = resolve_task_config(TASK, None, overrides=["physics=newton_mjwarp"])
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.video_recorders = []
    cfg.sim.visualizer_cfgs = []
    # Flat tiles for this diagnostic. Keep the column-to-motion mapping.
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = {
        name: MeshPlaneTerrainCfg(proportion=tile.proportion)
        for name, tile in gen.sub_terrains.items()
    }
    gen.use_cache = False
    if hasattr(cfg.events, "push_robot"):
        cfg.events.push_robot = None

    with launch_simulation(cfg):
        env = gym.make(TASK, cfg=cfg)
        try:
            with torch.inference_mode():
                summary = record(env, out)
        finally:
            env.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary), flush=True)
    print("OUTPUT:", out, flush=True)
    if summary["fixed_sample"] is None:
        print("LOGGING_OK_NO_DOUBLE_SUPPORT: no valid frame had both feet above 5 N.")
    else:
        print("FOOT_LOAD_TARGET_LOGGING_AND_SPLITS_OK")


if __name__ == "__main__":
    main()
