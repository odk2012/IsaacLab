Changed
^^^^^^^

* Removed stance-based tracking suppression and the six standing-only reward terms
  from the G1 rigid mimic task, and added the existing contact-gated foot sliding
  penalty with weight -0.25. Tracking reward weights and the adaptive assistance
  schedule were preserved.

Deprecated
^^^^^^^^^^

* Deprecated and ignored the legacy stance configuration fields. Compatibility
  weights now remain at zero for standing and one for tracking. Existing
  phase-aware soft-terrain action/event consumers consequently no longer enter
  stance mode; this experiment was scoped to rigid-ground walking.
