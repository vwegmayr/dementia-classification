Experiments
============

Preprocessing
----------------

Given raw images in NIFTI format as input, this module will skull-strip the
images, register it to a study-specific template brain,
apply gaussian smoothing, subsample the image and augment the data by rotation
and translation.

Baseline
-----------

Given the preprocessed images, this module would run the baseline 3D CNN model.


