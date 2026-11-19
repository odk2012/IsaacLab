# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain with one column per motion clip, each tile holding that clip's terrain boxes."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_goalconditioned_perception.mdp.terrains import (
    MotionStairsTerrainCfg,
)


def motion_terrain_generator(motion_files: Sequence[str], num_rows: int = 1, size: float = 10.0) -> TerrainGeneratorCfg:
    """A curriculum-ordered generator whose column ``i`` holds the boxes of ``motion_files[i]``.

    The tile origin is the clip frame's origin, with the boxes centered in the tile, so clip poses map to the world by
    adding the environment origin. Rows are identical copies.
    """
    return TerrainGeneratorCfg(
        seed=42,
        size=(size, size),
        num_rows=num_rows,
        num_cols=len(motion_files),
        border_width=5.0,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        curriculum=True,  # columns follow the sub-terrain order
        use_cache=False,
        sub_terrains={
            f"clip_{i:02d}": MotionStairsTerrainCfg(proportion=1.0, motion_file=path)
            for i, path in enumerate(motion_files)
        },
    )
