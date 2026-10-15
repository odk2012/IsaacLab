"""RTX2000 jogging-only training from scratch on the matched multi-mimic runtime."""
import argparse
import copy
from collections import deque
from datetime import datetime
import getpass
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HOME = Path.home()
HERE = Path(__file__).resolve().parent
ITERATIONS = 30000
TERM = "motion_feet_lin_vel"
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
MOTION = HOME / "lafan_jog_runs/run1_subject2_95to101s_20261001-152600-337650/conversion/run1_subject2_20261001-152600-486759/motion_60fps.npz"
MOTION_SHA = "1413a5df83f91803d2ef55945002152598648424308e0c8ad1db63f9ae94a237"
LATEST = HOME / "jogging-foot-velocity-latest.txt"
EXPECTED_REWARD = {
    "func": "isaaclab_tasks.contrib.mimic.mdp.rewards:motion_global_body_linear_velocity_error_exp",
    "weight": "1.0", "params": {"command_name": "motion", "std": "1.0", "body_names": FEET},
}


def initialize():
    global ctl
    spec = importlib.util.spec_from_file_location("matched_control", HOME / "matched-walking-control/run.py")
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    for name in ("runtime_check", "check_payload", "verify_worktree", "WORKS", "save", "load_config", "PYTHON"):
        globals()[name] = getattr(ctl, name)


def validate_inputs():
    runtime = runtime_check()
    check_payload()
    verify_worktree("multi")
    if not MOTION.is_file() or ctl.sha(MOTION) != MOTION_SHA:
        raise RuntimeError("Reviewed jogging NPZ missing or changed")
    return runtime


def compare_configs(env, agent, smoke, variant):
    env, agent = copy.deepcopy(env), copy.deepcopy(agent)
    actual = env["rewards"].pop(TERM, None)
    errors = ctl.differences(EXPECTED_REWARD, actual, "/env/rewards/" + TERM)
    motion = env["commands"]["motion"]
    for key, expected, historical in (
        ("motion_file", str(MOTION), str(ctl.MOTION)),
        ("motion_files", [str(MOTION)], [str(ctl.MOTION)]),
    ):
        errors += ctl.differences(expected, motion.get(key), "/env/commands/motion/" + key)
        motion[key] = historical
    errors += ctl.differences("5" if smoke else str(ITERATIONS), agent.get("max_iterations"), "/agent/max_iterations")
    agent["max_iterations"] = "5" if smoke else str(ctl.ITERATIONS)
    return errors + ctl.compare_configs(env, agent, smoke, variant)


def install_checkpoint_hooks(run, smoke):
    from rsl_rl.runners import OnPolicyRunner
    original_save, original_learn = OnPolicyRunner.save, OnPolicyRunner.learn

    def no_load(*args, **kwargs):
        raise RuntimeError("Fresh jogging experiment must not load a checkpoint")

    def fresh_learn(runner, *args, **kwargs):
        if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
            raise RuntimeError("Expected fresh policy and empty optimizer state")
        stage = "smoke-multi" if smoke else "training"
        save(run / (stage + "-fresh-start.json"), {
            "initial_iteration": 0, "optimizer_state_empty": True, "checkpoint_loaded": False,
        })
        return original_learn(runner, *args, **kwargs)

    def checkpoint_save(runner, path, infos=None):
        cmd = runner.env.unwrapped.command_manager.get_term("motion")
        fields = ("bin_failed_count", "_current_bin_failed", "bin_start_failed_count", "bin_start_count",
                  "_current_start_failed", "_current_start_count", "assist_scale", "start_bins",
                  "_start_bin_valid", "_resample_pending", "time_steps")
        extra = {} if infos is None else dict(infos)
        extra["motion_curriculum_snapshot"] = {
            "schema_version": 1,
            "motion_sha256": MOTION_SHA,
            "state": {k: getattr(cmd, k).detach().cpu().clone() for k in fields},
            "note": "Statistics snapshot for future recovery; not full simulator/RNG state. Fresh jogging run.",
        }
        return original_save(runner, path, extra)

    OnPolicyRunner.load = no_load
    OnPolicyRunner.learn = fresh_learn
    OnPolicyRunner.save = checkpoint_save


def backend(run, variant, smoke):
    runtime_check()
    check_payload()
    verify_worktree(variant)
    work = WORKS[variant]
    os.chdir(work)
    sys.path[:0] = [str(p) for p in sorted((work / "source").iterdir()) if (p / p.name).is_dir()]
    import warp as wp
    wp.config.enable_backward = False
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    from isaaclab_rl.entrypoints.backends import train_rsl_rl as train
    from isaaclab.managers import RewardTermCfg
    from isaaclab_tasks.contrib.mimic.mdp.rewards import motion_global_body_linear_velocity_error_exp
    validate_inputs()
    install_checkpoint_hooks(run, smoke)

    stage = f"smoke-{variant}" if smoke else "training"
    original_dump = train.dump_train_configs
    original_create = train.create_isaaclab_env
    state = {"config_checked": False}

    def checked_create(*args, **kwargs):
        cfg = args[1]
        motion_cfg = cfg.commands.motion
        if motion_cfg.motion_file != str(MOTION) or motion_cfg.stance_phase_ranges:
            raise RuntimeError("Incorrect single jogging reference or stance setup.")
        if variant == "multi":
            if motion_cfg.motion_files != [str(MOTION)] or motion_cfg.motion_stance_phase_ranges != [[]]:
                raise RuntimeError("This diagnostic must have exactly one jogging clip.")
        if cfg.rewards.feet_slide.weight != -0.25:
            raise RuntimeError("feet_slide weight changed.")
        if any(k.startswith("standing_") and v is not None for k, v in vars(cfg.rewards).items()):
            raise RuntimeError("Standing-only rewards must remain disabled.")
        if getattr(cfg.rewards, TERM, None) is not None:
            raise RuntimeError("Experimental reward already exists")
        if not set(FEET).issubset(motion_cfg.body_names):
            raise RuntimeError("Both ankle links must be present in tracked bodies")
        setattr(cfg.rewards, TERM, RewardTermCfg(
            func=motion_global_body_linear_velocity_error_exp, weight=1.0,
            params={"command_name": "motion", "std": 1.0, "body_names": FEET},
        ))
        env = original_create(*args, **kwargs)
        base = env.unwrapped
        cmd = base.command_manager.get_term("motion")
        if not torch.all(cmd.tracking_weight == 1) or torch.count_nonzero(cmd.frame_stance_weight):
            raise RuntimeError("Stance logic remains active.")
        if variant == "multi" and cmd.num_motions != 1:
            raise RuntimeError("Expected a one-clip MotionLibrary.")
        if base.num_envs != 2048:
            raise RuntimeError("Environment count differs from the historical run.")
        reward = base.reward_manager.get_term_cfg(TERM)
        if reward.weight != 1.0 or reward.params["body_names"] != FEET:
            raise RuntimeError("Wrong ankle velocity reward")
        value = reward.func(base, **reward.params)
        if value.shape != (2048,) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise RuntimeError("Invalid ankle velocity reward values")
        term = base.reward_manager.get_term_cfg("feet_slide")
        robot = base.scene["robot"]
        sensor = base.scene.sensors["contact_forces"]
        feet = [robot.body_names[i] for i in term.params["asset_cfg"].body_ids]
        sensed = [sensor.body_names[i] for i in term.params["sensor_cfg"].body_ids]
        if feet != sensed or set(feet) != {"left_ankle_roll_link", "right_ankle_roll_link"}:
            raise RuntimeError("Foot-sliding reward does not target the same two ankles.")
        save(run / f"{stage}-runtime-check.json", {
            "variant": variant, "feet": feet, "feet_slide_weight": -0.25,
            "tracking_all_frames": True, "jogging_reference": str(MOTION),
            "num_envs": base.num_envs, "assistance_enabled_during_training": cfg.events.assistive_wrench is not None,
        })
        return env

    def checked_dump(log_dir, env_cfg, agent_cfg):
        original_dump(log_dir, env_cfg, agent_cfg)
        params = Path(log_dir) / "params"
        errors = compare_configs(load_config(params / "env.yaml"), load_config(params / "agent.yaml"), smoke, variant)
        wrong_imports = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("isaaclab") and path and not Path(path).resolve().is_relative_to(work.resolve()):
                wrong_imports[name] = path
        save(run / f"{stage}-config-check.json", {"passed": not errors and not wrong_imports, "differences": errors, "wrong_imports": wrong_imports})
        if errors or wrong_imports:
            raise RuntimeError(f"Historical config/import mismatch; learning prevented. See {stage}-config-check.json")
        for name in ("env.yaml", "agent.yaml"):
            shutil.copyfile(params / name, run / f"{stage}-{name}")
        save(run / f"{stage}-output.json", {"log_dir": log_dir})
        state["config_checked"] = True
        print("CONFIG_OK: only ankle velocity reward added; other historical settings matched", flush=True)

    train.create_isaaclab_env = checked_create
    train.dump_train_configs = checked_dump
    args = [
        "--task", "IsaacContrib-Mimic-G1-29dof-v0", "--num_envs", "2048",
        "--max_iterations", "5" if smoke else str(ITERATIONS), "--seed", "42",
        "--run_name", run.name + "-" + stage,
        "--logger", "tensorboard" if smoke else "wandb", "--viz", "none",
        "--log_project_name", "g1-lafan-mimic", "physics=newton_mjwarp",
        "agent.experiment_name=matched_jogging",
        "agent.wandb_project=g1-lafan-mimic",
        "agent.neptune_project=g1-lafan-mimic",
        "env.commands.motion.motion_file=" + json.dumps(str(MOTION)),
        "env.commands.motion.stance_phase_ranges=[]",
        "env.commands.motion.debug_vis=false", "env.video_recorders=[]",
    ]
    if variant == "multi":
        args.extend([
            "env.commands.motion.motion_files=" + json.dumps([str(MOTION)]),
            "env.commands.motion.motion_stance_phase_ranges=[[]]",
        ])
    save(run / f"{stage}-command.json", args)
    train.run(args)
    if not state["config_checked"]:
        raise RuntimeError("Training returned without reaching the config gate.")
    output = json.loads((run / f"{stage}-output.json").read_text())
    checkpoint = Path(output["log_dir"]) / ("model_4.pt" if smoke else "model_29999.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Expected final checkpoint missing: " + str(checkpoint))
    save(run / f"{stage}-completed.json", {"checkpoint": str(checkpoint), "iterations": 5 if smoke else ITERATIONS, "checkpoint_loaded": False})



def worker(run):
    import fcntl
    with (HOME / ".matched-walking-training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for smoke in (True, False):
            ctl.gpu_idle()
            stage = "smoke-multi" if smoke else "training"
            save(run / "status.json", {"stage": stage, "pid": os.getpid()})
            env = dict(os.environ)
            for key in list(env):
                if key.startswith("WANDB_") and key not in {"WANDB_API_KEY", "WANDB_BASE_URL"}:
                    env.pop(key)
            env.update(WANDB_ENTITY="jnskkmr", WANDB_PROJECT="g1-lafan-mimic", WANDB_MODE="disabled" if smoke else "online")
            if not smoke:
                env.update(WANDB_RUN_ID=run.name, WANDB_RESUME="never", WANDB_NAME=run.name,
                    WANDB_NOTES="Fresh jogging-only training in matched multi code, 2048 envs, 30000 updates; ankle velocity reward weight=1,std=1; feet_slide=-0.25; no walking checkpoint, no transitions.")
            cmd = [str(PYTHON), "-u", str(HERE / "run.py"), "backend", "--run", str(run)]
            if smoke:
                cmd.append("--smoke")
            with (run / f"{stage}.log").open("w") as log:
                subprocess.run(cmd, cwd=WORKS["multi"], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200 if smoke else None)
        result = json.loads((run / "training-completed.json").read_text())
        save(run / "status.json", {"stage": "completed", **result,
             "wandb_url": "https://wandb.ai/jnskkmr/g1-lafan-mimic/runs/" + run.name})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "worker", "backend"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    initialize()
    if args.action == "status":
        run = args.run or Path(LATEST.read_text().strip())
        print("RUN:", run)
        print((run / "status.json").read_text())
        for name in ("launch.log", "smoke-multi.log", "training.log"):
            if (run / name).exists():
                with (run / name).open(errors="replace") as stream:
                    print(name + ":\n" + "".join(deque(stream, maxlen=18)))
        return
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTORCH_JIT", "DISPLAY"):
        os.environ.pop(key, None)
    os.environ.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", HYDRA_FULL_ERROR="1")
    if args.action == "backend":
        backend(args.run, "multi", args.smoke)
        return
    if args.action == "worker":
        try:
            worker(args.run)
        except BaseException as exc:
            save(args.run / "status.json", {"stage": "failed", "error": str(exc)})
            raise
        return
    runtime = validate_inputs()
    ctl.gpu_idle()
    import wandb
    key = os.environ.get("WANDB_API_KEY") or getpass.getpass("Your W&B API key (hidden): ").strip()
    if not key or any(c.isspace() for c in key):
        raise RuntimeError("Invalid API key input")
    os.environ["WANDB_API_KEY"] = key
    api = wandb.Api(api_key=key)
    if api.viewer.username != "odk2000201":
        raise RuntimeError("Use your own W&B credentials: expected odk2000201")
    if api.project("g1-lafan-mimic", entity="jnskkmr") is None:
        raise RuntimeError("Cannot access W&B project")
    run = HOME / "jogging_foot_velocity_runs" / ("jog-footvel-fresh30k-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    run.mkdir(parents=True)
    shutil.copy2(HERE / "run.py", run / "experiment-source.py")
    save(run / "runtime.json", runtime)
    (run / "gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    save(run / "experiment.json", {
        "motion": str(MOTION), "motion_sha256": MOTION_SHA,
        "source_contract": ctl.SPEC, "new_reward": EXPECTED_REWARD,
        "num_envs": 2048, "iterations": ITERATIONS, "seed": 42,
        "checkpoint_loaded": False, "skills": ["jogging"], "transition_training": False,
        "note": "Same matched multi code/runtime and foot-velocity/feet-slide settings. Jogging only, from scratch. Smoke and full training initialize independent fresh models. Source contract motion hash refers to historical WALKING baseline; actual jogging input is recorded above.",
    })
    save(run / "status.json", {"stage": "queued"})
    with (run / "launch.log").open("w") as log:
        process = subprocess.Popen([str(PYTHON), "-u", str(HERE / "run.py"), "worker", "--run", str(run)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ))
    LATEST.write_text(str(run) + "\n")
    print("BACKGROUND_LAUNCHED:", process.pid, flush=True)
    print("RUN:", run, flush=True)
    print("Five-update smoke first; then FRESH JOGGING-ONLY 2048-env / 30000-update training. No checkpoint loaded.", flush=True)


if __name__ == "__main__":
    main()
