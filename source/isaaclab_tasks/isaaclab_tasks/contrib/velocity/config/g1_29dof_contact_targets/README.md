# G1 contact targets (rigid)

`IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets` (and `-Play`) trains the 29-DoF G1 to follow the foot and
hand contacts of motion clips, on rigid terrain and without perception. It reuses the
[rigid task](../g1_29dof_rigid/README.md)'s robot, actions, physics, randomization, and rough-terrain rewards.

## Clips and terrain

Nine clips. Seven train, drawn per episode with stairs and walking in equal shares (`MOTION_WEIGHTS` in
`env_cfg.py`); two are held out as a test set (`TEST_MOTION_FILES`), never drawn in training and the only clips in
`-Play`:

| Clip | Content | Use |
| --- | --- | --- |
| `motion_yaw_000`, `_090`, `_180` | Pyramid stairs of the goal-conditioned task, approached from three sides, 7.5 s | Train |
| `motion_yaw_270` | The same pyramid climb from the fourth side | Test |
| `kimodo_ramp_seed45`, `_seed46` | Walk up a 0.1 m ramp, 6 s ([`data/motion_library`](data/motion_library/README.md)) | Train |
| `kimodo_ramp_seed47` | Walk up the ramp with a different gait (more, shorter steps) | Test |
| `kimodo_platform_seed44`, `_seed45` | Walk onto a 0.15 m platform, 4 s | Train |

The terrain has one column per clip, holding that clip's boxes; at reset each robot moves onto its clip's column
and starts on the clip's pose at a random frame (the first frame in `-Play`), with ±0.05 rad joint noise. The clip
then advances one frame per step, and the episode ends at the clip's last frame, at a fall, or when the root is
1 m from the clip's root.

## Contacts

Contacts are measured from each clip's geometry when it is loaded (`mdp/motion.py`). A foot is in contact when at
least two corners of its foot box are within 1 cm of the terrain and its ankle moves slower than 0.3 m/s (dragged
swing feet do not count); a hand when its sphere is within 1 cm and slower than 0.3 m/s. Contacts and gaps shorter
than three frames are dropped. On the clips that carry contact labels this agrees with them on 84% (stairs) and
89–94% (platform) of the frames. None of the nine clips uses the hands.

## Observations

The actor reads the rigid task's proprioception (no height scan) and, in place of the velocity command, the
`command` group: for the left foot, right foot, left hand, and right hand, the position of the current or next
contact relative to the root in its yaw frame, whether the clip has it in contact, and the time until that contact
begins as a fraction of 2 s (1 when no contact is announced within 2 s). The contact target is the mean ankle roll
link (or hand sphere) position over the contact. The critic reads the same group with the rigid task's critic and
privileged observations. The left/right mirror swaps the bodies and negates the lateral coordinate.

## Rewards

The rigid task's rough-terrain rewards without `track_lin_vel_xy`, `track_ang_vel_z`, and `track_heading`. The
terms that tell standing from walking (`joint_deviation`, `feet_air_time`, `break_contact`, `no_fly`,
`contact_impulse`, `standing_foot_motion`) read the clip's root velocity as their command. `foot_contact_target`
(weight 3) pays, averaged over the feet the clip has in contact, `exp(-|p_foot - p_target|² / 0.15²)` of the robot's
ankle roll link to the clip's contact position. The curriculum is empty.

## Training

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
    --task IsaacContrib-Velocity-G1-29dof-Rigid-ContactTargets --num_envs 4096 --headless
```

Runs log to `g1_29dof_rigid_contact_targets_ppo`.
