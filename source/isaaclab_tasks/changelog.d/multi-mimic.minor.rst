Added
^^^^^

* Added multiple reference files to the humanoid mimic motion command, with uniform
  clip selection at episode reset and independent clip clocks, stance ranges,
  adaptive sampling statistics, and assistive-wrench acceleration boundaries.
  Existing ``motion_file`` configurations remained supported; ``motion_files``
  took precedence when populated. Reference clips were required to match the control rate.

Fixed
^^^^^

* Fixed mimic reset events and command resampling for slice environment indices.
