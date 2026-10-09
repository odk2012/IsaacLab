# LAFAN walking: continuous tracking and foot sliding penalty

This branch preserves the code used for the 30,000-update rigid-ground walking
experiment, based on `e1f1d6d2a805d88f37b38ea418462ab5a2766808`.
The five modified Python files preserve the experiment's source code. Repository
formatting normalized blank lines in ``commands_cfg.py``; its Python syntax tree
is unchanged from the recorded training source.
It is a single-motion G1 experiment. Soft-terrain consumers of the shared stance
properties were not validated with these changes.

## Training changes

- Removed predefined stance intervals and tracking reward phase multipliers.
- Removed the six standing-only reward terms from the rigid task configuration.
- Retained deprecated stance fields and properties for import compatibility;
  standing weights are always zero and tracking weights are always one.
- Added the existing `core.velocity.mdp.feet_slide` penalty with weight `-0.25`
  for both ankle-roll links. It uses contact-force history and horizontal link
  speed; it is a contact-gated proxy rather than a sole-contact slip measurement.
- Updated an assistance docstring. The assistance curriculum was unchanged.

PPO settings, symmetry augmentation, mirror loss, training randomization,
termination conditions, and other reward weights were inherited from the base.
Training started from scratch with 2,048 environments, 30,000 PPO updates, and
seed 42. A 32-environment, five-update smoke check ran before the full experiment.

## Reference motion

The experiment used `walk1_subject1.csv` from the G1
[LAFAN1 retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
source frames 121 through 301 inclusive (1-based, source time 4–10 seconds).
The converted NPZ contains 360 frames at 60 Hz, named joints and bodies, and
explicit `xyzw` quaternion metadata.

The exact training NPZ SHA-256 is:

```text
e06aa535399f1da1ceeb2fb0b82579d611a0b71240bcdbf4f343a09d8d6fcbb1
```

Motion data is supplied separately. The conversion utility is maintained in the
[LAFAN conversion branch](https://github.com/odk2012/IsaacLab/tree/codex/lafan-walk-reproduction/source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/utils).
Use a motion converted for this task's robot configuration. Regeneration with
another code revision or robot asset is not an exact reproduction of this run.

## Run training

Use a Linux checkout of this branch with its own configured Newton environment.
Install the locked `mimic` and `rsl-rl` extras and make W&B available before
running. Authenticate W&B locally; no credentials belong in the repository.
Run from the repository root:

```bash
read -r -p "Absolute path to the walking NPZ: " motion_file
test -f "$motion_file" || exit 1
printf '%s  %s\n' \
  e06aa535399f1da1ceeb2fb0b82579d611a0b71240bcdbf4f343a09d8d6fcbb1 \
  "$motion_file" | sha256sum --check || exit 1

export WANDB_ENTITY=jnskkmr
export WANDB_PROJECT=g1-lafan-mimic
export WANDB_MODE=online

uv run --frozen --no-sync isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-v0 \
  --num_envs 2048 --max_iterations 30000 --seed 42 \
  --run_name lafan_walk_continuous_tracking_feet_slide \
  --logger wandb --log_project_name g1-lafan-mimic --viz none \
  physics=newton_mjwarp \
  "env.commands.motion.motion_file=\"$motion_file\"" \
  'env.video_recorders=[]' agent.experiment_name=lafan_walk
```

Choose a W&B entity/project you can write to if reproducing outside the lab.
There is no resume or checkpoint argument. Assistance remains enabled during
training according to the existing curriculum. For a smoke check, use 32
environments and five updates, then start the full run separately.

The original job used the existing Python environment with this worktree's source
packages prepended to its import paths. Its launcher checked the imported paths,
reward set, sensor/link mapping, and saved PPO configuration before training.
The command above uses the repository CLI with those same training settings;
it does not reproduce the machine-specific launcher or detached-process wrapper.

For policy playback, explicitly set `env.events.assistive_wrench=null` if an
assistance-free comparison is intended. The PLAY task relaxes some training
termination conditions, so completing a PLAY rollout is not a training-mode
success-rate evaluation.
