# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion clips with foot and hand contacts as targets, and their observation, reward, and reset terms."""

from .command import MotionContactCommand, MotionContactCommandCfg
from .events import reset_to_motion
from .motion import CONTACT_BODIES, ContactMotionLibrary
from .observations import mirror_upcoming_contacts, upcoming_contacts_b
from .rewards import foot_contact_target_exp
from .terminations import motion_end, root_far_from_clip
from .terrains import motion_terrain_generator
