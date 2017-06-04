from os import path
import os
import pickle
import re
import argparse
import sys
import nibabel as nb
from scipy.ndimage.interpolation import shift
import numpy as np

from dementia_prediction.config_wrapper import Config
from dementia_prediction.cnn_baseline.data_input import DataInput
from dementia_prediction.cnn_baseline.baseline import CNN

config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))


paths = config.config.get('data_paths')
#regex = r"-T1_brain_sub_rotation5\.nii\.gz$"
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)
print(patients_dict)
train_filenames = []
train_labels = []
valid_filenames = []
valid_labels = []
s_codes = []
p_codes = []
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-T1_brain_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-T1_brain_subsampled.nii.gz')
            output = pat_code[0]+'-T1_translated_x_affine_corrected.nii.gz'
            if not os.path.exists(output):
                mri_image = nb.load(input_file)
                aff = mri_image.get_affine()
                mri_image = mri_image.get_data()
                translated_image_x = shift(mri_image, [4, 0, 0], mode='nearest')
                im = nb.Nifti1Image(translated_image_x, affine=aff)
                nb.save(im, output)
                print("Saving to "+output)
            else:
                print("Exists"+output)

            output = pat_code[0]+'-T1_translated_y_affine_corrected.nii.gz'
            if not os.path.exists(output):
                translated_image_y = shift(mri_image, [0, 4, 0], mode='nearest')
                im = nb.Nifti1Image(translated_image_y, affine=aff)
                print("Saving to "+output)
                nb.save(im, output)
            else:
                print("Exists"+output)

            output = pat_code[0]+'-T1_translated_z_affine_corrected.nii.gz'
            if not os.path.exists(output):
                translated_image_z = shift(mri_image, [0, 0, 4], mode='nearest')
                im = nb.Nifti1Image(translated_image_z, affine=aff)
                print("Saving to "+output)
                nb.save(im, output)
            else:
                print("Exists"+output)

