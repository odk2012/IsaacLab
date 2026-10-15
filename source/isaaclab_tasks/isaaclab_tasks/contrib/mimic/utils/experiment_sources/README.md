# Original experiment sources

Snapshots of the scripts and configuration used for the matched experiments.
The launchers add motion_feet_lin_vel at runtime: weight=1.0, std=1.0, targeting both ankle roll links. The repository includes feet_slide=-0.25 and disabled stance/standing rewards.

Walking: 15k initial updates followed by 15k with foot velocity tracking.
Jogging: fresh 30k updates with foot velocity tracking.

These original scripts retain lab-specific paths and dependency checks. They expect these three directories under the user's home directory. Motion data and the walking continuation checkpoint are required separately.
No checkpoints, videos, credentials, or new transition implementation are included.
