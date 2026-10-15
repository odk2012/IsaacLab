"""Train walking with the audited multi command on the historical single runtime.

Linux/RTX2000 only. No package installation, existing checkout edits, or resumes.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
from datetime import datetime
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
HOME = Path.home()
REPO = HOME / "IsaacLab"
PYTHON = REPO / ".venv/bin/python"
LATEST = HOME / "matched-walking-latest.txt"
ITERATIONS = 15000
SPEC = json.loads((HERE / "manifest.json").read_text())
PREFIX = "source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/"
WORKS = {v: HOME / f"IsaacLab-matched-walking-{v}" for v in ("single", "multi")}
MOTION = HOME / SPEC["motion_relative"]


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).strip()


def expected_files(variant):
    files = dict(SPEC["single_files"])
    if variant == "multi":
        files.update(SPEC["multi_files"])
    return files


def runtime_check():
    if platform.system() != "Linux":
        raise RuntimeError("Run on the RTX2000 Linux laptop, not the local Windows computer.")
    if Path(sys.executable).absolute().parent != PYTHON.parent:
        raise RuntimeError(f"Use the historical single environment: {PYTHON}")
    if platform.python_version() != SPEC["python_version"]:
        raise RuntimeError("Historical Python version no longer matches. No training started.")
    versions = {name: importlib.metadata.version(name) for name in SPEC["versions"]}
    differences = {k: [SPEC["versions"][k], v] for k, v in versions.items() if SPEC["versions"][k] != v}
    if differences:
        raise RuntimeError("Historical runtime changed; do not auto-upgrade/downgrade: " + json.dumps(differences))
    return {"python": sys.version, "executable": sys.executable, "packages": versions}


def gpu_idle():
    busy = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader",
    ], text=True).strip()
    if busy:
        raise RuntimeError("GPU compute is already active; stop here without touching it:\n" + busy)


def check_payload():
    for entry in [*SPEC["single_files"].values(), *SPEC["multi_files"].values()]:
        if sha(HERE / entry["payload"]) != entry["sha256"]:
            raise RuntimeError("Package checksum mismatch: " + entry["payload"])
    for name, digest in SPEC["saved_configs"].items():
        if sha(HERE / "baseline" / name) != digest:
            raise RuntimeError("Historical config checksum mismatch: " + name)
    if sha(MOTION) != SPEC["motion_sha256"]:
        raise RuntimeError("Walking NPZ differs from the historical reference.")


def verify_worktree(variant):
    work = WORKS[variant]
    files = expected_files(variant)
    if git(work, "rev-parse", "HEAD") != SPEC["base"]:
        raise RuntimeError(f"Unexpected base: {work}")
    if git(work, "branch", "--show-current") != f"codex/matched-walking-{variant}":
        raise RuntimeError(f"Unexpected branch: {work}")
    changed = set(git(work, "diff", "--name-only", "HEAD").splitlines())
    if changed - files.keys():
        raise RuntimeError(f"Unrelated tracked edits in {work}: {changed - files.keys()}")
    for relative, entry in files.items():
        if sha(work / relative) != entry["sha256"]:
            raise RuntimeError("Experiment source has changed: " + str(work / relative))
    # Untracked Python/asset files could shadow or alter the controlled source.
    extra = git(work, "ls-files", "--others", "--exclude-standard", "--", "source").splitlines()
    if set(extra) - files.keys():
        raise RuntimeError(f"Unexpected untracked source: {extra}")


def prepare():
    runtime = runtime_check()
    check_payload()
    # Match the multi implementation that was actually audited, not a moving branch.
    multi = HOME / "IsaacLab-multi-mimic"
    for relative, entry in SPEC["multi_files"].items():
        content = ast.dump(ast.parse((multi / relative).read_text()))
        if hashlib.sha256(content.encode()).hexdigest() != entry["ast_sha256"]:
            raise RuntimeError("Multi implementation changed since audit: " + relative)
    for variant, work in WORKS.items():
        if not work.exists():
            subprocess.run([
                "git", "-C", str(REPO), "worktree", "add", "-b",
                f"codex/matched-walking-{variant}", str(work), SPEC["base"],
            ], check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            # Only initialize a newly created checkout. Never overwrite an existing one.
            for relative, entry in expected_files(variant).items():
                target = work / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE / entry["payload"], target)
        verify_worktree(variant)
    return runtime


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append({"path": path + "/" + key, "expected": a.get(key, "<missing>"), "actual": b.get(key, "<missing>")})
            else:
                result.extend(differences(a[key], b[key], path + "/" + key))
        return result
    if a != b:
        return [{"path": path, "expected": a, "actual": b}]
    return []


def load_config(path):
    import yaml
    # YAML BaseLoader does not execute Python tags. Both sides retain the same
    # scalar representation, including tuple/slice encodings in IsaacLab dumps.
    return yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        for root in [HOME / "IsaacLab-walking-rewards", *WORKS.values()]:
            value = value.replace(str(root) + "/", "$SOURCE/")
        return value
    return value


def compare_configs(env, agent, smoke, variant):
    env = normalize(env)
    agent = normalize(agent)
    expected_env = normalize(load_config(HERE / "baseline/env.yaml"))
    expected_agent = normalize(load_config(HERE / "baseline/agent.yaml"))
    # Exclusions are deliberately narrow and reported in the manifest.
    for data in (env, expected_env):
        data.pop("log_dir", None)
    for data in (agent, expected_agent):
        for key in ("run_name", "experiment_name", "logger"):
            data.pop(key, None)
    expected_agent["max_iterations"] = "5" if smoke else str(ITERATIONS)
    if variant == "multi":
        expected_env["commands"]["motion"]["motion_files"] = [str(MOTION)]
        expected_env["commands"]["motion"]["motion_stance_phase_ranges"] = [[]]
    return differences(expected_env, env, "/env") + differences(expected_agent, agent, "/agent")


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

    stage = f"smoke-{variant}" if smoke else "training"
    original_dump = train.dump_train_configs
    original_create = train.create_isaaclab_env
    state = {"config_checked": False}

    def checked_create(*args, **kwargs):
        cfg = args[1]
        motion_cfg = cfg.commands.motion
        if motion_cfg.motion_file != str(MOTION) or motion_cfg.stance_phase_ranges:
            raise RuntimeError("Incorrect single walking reference or stance setup.")
        if variant == "multi":
            if motion_cfg.motion_files != [str(MOTION)] or motion_cfg.motion_stance_phase_ranges != [[]]:
                raise RuntimeError("This diagnostic must have exactly one walking clip.")
        if cfg.rewards.feet_slide.weight != -0.25:
            raise RuntimeError("feet_slide weight changed.")
        if any(k.startswith("standing_") and v is not None for k, v in vars(cfg.rewards).items()):
            raise RuntimeError("Standing-only rewards must remain disabled.")
        env = original_create(*args, **kwargs)
        base = env.unwrapped
        cmd = base.command_manager.get_term("motion")
        if not torch.all(cmd.tracking_weight == 1) or torch.count_nonzero(cmd.frame_stance_weight):
            raise RuntimeError("Stance logic remains active.")
        if variant == "multi" and cmd.num_motions != 1:
            raise RuntimeError("Expected a one-clip MotionLibrary.")
        if base.num_envs != 2048:
            raise RuntimeError("Environment count differs from the historical run.")
        term = base.reward_manager.get_term_cfg("feet_slide")
        robot = base.scene["robot"]
        sensor = base.scene.sensors["contact_forces"]
        feet = [robot.body_names[i] for i in term.params["asset_cfg"].body_ids]
        sensed = [sensor.body_names[i] for i in term.params["sensor_cfg"].body_ids]
        if feet != sensed or set(feet) != {"left_ankle_roll_link", "right_ankle_roll_link"}:
            raise RuntimeError("Foot-sliding reward does not target the same two ankles.")
        save(run / f"{stage}-runtime-check.json", {
            "variant": variant, "feet": feet, "feet_slide_weight": -0.25,
            "tracking_all_frames": True, "walking_reference": str(MOTION),
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
        print("HISTORICAL_SETTINGS_MATCH_OK: rewards, observations, reset, assistance, physics, PPO and symmetry", flush=True)

    train.create_isaaclab_env = checked_create
    train.dump_train_configs = checked_dump
    args = [
        "--task", "IsaacContrib-Mimic-G1-29dof-v0", "--num_envs", "2048",
        "--max_iterations", "5" if smoke else str(ITERATIONS), "--seed", "42",
        "--run_name", run.name + "-" + stage,
        "--logger", "tensorboard" if smoke else "wandb", "--viz", "none",
        "--log_project_name", "g1-lafan-mimic", "physics=newton_mjwarp",
        "agent.experiment_name=matched_walking",
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
    checkpoint = Path(output["log_dir"]) / ("model_4.pt" if smoke else f"model_{ITERATIONS - 1}.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Expected final checkpoint missing: " + str(checkpoint))
    save(run / f"{stage}-completed.json", {"checkpoint": str(checkpoint), "iterations": 5 if smoke else ITERATIONS})


def worker(run):
    import fcntl
    with (HOME / ".matched-walking-training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for variant, smoke in [("single", True), ("multi", True), ("multi", False)]:
            gpu_idle()
            stage = f"smoke-{variant}" if smoke else "training"
            save(run / "status.json", {"stage": stage, "pid": os.getpid()})
            env = dict(os.environ)
            # Never publish smoke runs or reuse a previous W&B run ID.
            for key in list(env):
                if key.startswith("WANDB_") and key not in {"WANDB_API_KEY", "WANDB_BASE_URL"}:
                    env.pop(key)
            env.update(WANDB_ENTITY="jnskkmr", WANDB_PROJECT="g1-lafan-mimic", WANDB_MODE="disabled" if smoke else "online")
            if not smoke:
                env.update(WANDB_RUN_ID=run.name, WANDB_RESUME="never", WANDB_NAME=run.name)
            command = [str(PYTHON), "-u", "-X", "faulthandler", str(HERE / "run.py"), "backend", "--run", str(run), "--variant", variant]
            if smoke:
                command.append("--smoke")
            with (run / f"{stage}.log").open("w") as log:
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               env=env, check=True, timeout=1200 if smoke else None)
        result = json.loads((run / "training-completed.json").read_text())
        save(run / "status.json", {"stage": "completed", **result, "wandb_url": f"https://wandb.ai/jnskkmr/g1-lafan-mimic/runs/{run.name}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "start", "worker", "backend", "status"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--variant", choices=["single", "multi"], default="multi")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "status":
        run = args.run or Path(LATEST.read_text().strip())
        print("RUN:", run)
        print((run / "status.json").read_text())
        for name in ("launch.log", "smoke-single.log", "smoke-multi.log", "training.log"):
            if (run / name).exists():
                with (run / name).open(errors="replace") as stream:
                    print(name + ":\n" + "".join(deque(stream, maxlen=16)))
        return
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTORCH_JIT", "DISPLAY"):
        os.environ.pop(key, None)
    os.environ.update(PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", HYDRA_FULL_ERROR="1")
    if args.action == "backend":
        backend(args.run, args.variant, args.smoke)
        return
    if args.action == "worker":
        try:
            worker(args.run)
        except BaseException as exc:
            save(args.run / "status.json", {"stage": "failed", "error": str(exc), "pid": os.getpid()})
            raise
        return
    runtime = prepare()
    print("SOURCE_AND_RUNTIME_READY:", json.dumps(runtime), flush=True)
    if args.action == "prepare":
        return
    gpu_idle()
    import wandb
    # A shared Linux account may have another person's credentials in .netrc.
    key = getpass.getpass("Your W&B API key (hidden): ").strip()
    if not key or any(c.isspace() for c in key):
        raise RuntimeError("Paste only your API key at the hidden prompt.")
    os.environ["WANDB_API_KEY"] = key
    api = wandb.Api(api_key=key)
    user = api.viewer.username
    if user != "odk2000201":
        raise RuntimeError(f"Wrong W&B account: {user}; expected odk2000201.")
    project = api.project("g1-lafan-mimic", entity="jnskkmr")
    if project is None:
        raise RuntimeError("Cannot access the intended W&B project.")
    run = HOME / "matched_walking_runs" / ("multi-walk-oldbase-15k-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    run.mkdir(parents=True)
    save(run / "runtime.json", runtime)
    save(run / "experiment.json", {
        "base": SPEC["base"], "runtime": str(PYTHON), "motion": str(MOTION),
        "motion_sha256": SPEC["motion_sha256"], "full_training_variant": "multi",
        "num_envs": 2048, "iterations": ITERATIONS, "seed": 42, "resume": False,
        "new_smokes": ["single:2048env:5updates", "multi:2048env:5updates"],
        "historical_comparison": "single walking feet-slide Oct08, compare at the same 15000-update point",
        "limitations": "Current GPU driver is recorded, not downgraded to its historical version. Physics is nondeterministic. A historical comparison is not a new paired-seed experiment.",
        "source_contract": SPEC,
    })
    (run / "gpu.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    save(run / "status.json", {"stage": "queued"})
    with (run / "launch.log").open("w") as log:
        process = subprocess.Popen([str(PYTHON), "-u", str(HERE / "run.py"), "worker", "--run", str(run)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=dict(os.environ))
    (run / "launcher.pid").write_text(str(process.pid) + "\n")
    LATEST.write_text(str(run) + "\n")
    print("BACKGROUND_LAUNCHED:", process.pid, flush=True)
    print("RUN:", run, flush=True)
    print(f"Two five-update checks first, then fresh MULTI walking-only 2048-env / {ITERATIONS}-update training.", flush=True)
    print("Check status before closing SSH. Keep the RTX2000 awake and powered on.", flush=True)


if __name__ == "__main__":
    main()
