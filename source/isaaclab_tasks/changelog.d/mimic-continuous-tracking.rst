Changed
^^^^^^^

* Added an ankle sliding penalty with weight -0.25 to G1 mimic training and
  kept motion tracking active throughout each reference.
* Removed standing-only reward terms from the G1 mimic task.

Deprecated
^^^^^^^^^^

* Deprecated global and per-motion stance ranges and stance blending. Populated
  ranges were ignored with a warning; remove these settings from new configs.
  Compatibility standing weights remained zero and tracking weights one.
