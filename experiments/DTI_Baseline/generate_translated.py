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
        regex = r"-DTI_FA_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-DTI_FA_subsampled.nii.gz')
            output = pat_code[0]+'-DTI_FA_translated_x.nii.gz'
            mri_image = 0
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

            output = pat_code[0]+'-DTI_FA_translated_y.nii.gz'
            if not os.path.exists(output):
                translated_image_y = shift(mri_image, [0, 4, 0], mode='nearest')
                im = nb.Nifti1Image(translated_image_y, affine=aff)
                print("Saving to "+output)
                nb.save(im, output)
            else:
                print("Exists"+output)

            output = pat_code[0]+'-DTI_FA_translated_z.nii.gz'
            if not os.path.exists(output):
                translated_image_z = shift(mri_image, [0, 0, 4], mode='nearest')
                im = nb.Nifti1Image(translated_image_z, affine=aff)
                print("Saving to "+output)
                nb.save(im, output)
            else:
                print("Exists"+output)
            """
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patients_dict[patient_code] == 0:
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    p_codes.append(patient_code)
            """
'''
print("Total Number of Stable Patients: "+str(len(s_codes)))
print("Total Number of Progressive Patients: "+str(len(p_codes)))
#s_split = int(paths['validation_split']*len(s_codes))
#p_split = int(paths['validation_split']*len(p_codes))
# Divide the data into train, valid and test
#s_codes = s_codes[:10]
#p_codes = p_codes[:10]
s_split = 20
p_split = 20
train_patients = s_codes[s_split:]+p_codes[p_split:]
valid_patients = s_codes[:s_split]+p_codes[:p_split]
#train_patients = s_codes+p_codes
#valid_patients = s_codes+p_codes
print("Train:", train_patients)
print("Valid:", valid_patients)
print("Total Number of unique Train Patients: "+str(len(train_patients)))
print("Total Number of unique Valid Patients: "+str(len(valid_patients)))
"""
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_brain_sub_rotation_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-T1_brain_sub_rotation_{0}.nii.gz'
                                             .format(rotation))
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code in train_patients:
                    #print(input_file, patients_dict[patient_code])
                    train_filenames.append(input_file)
                    train_labels.append(patients_dict[patient_code])
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_brain_sub_rotation5_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-T1_brain_sub_rotation5_{0}.nii.gz'
                                             .format(rotation))
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code in train_patients:
                    #print(input_file, patients_dict[patient_code])
                    train_filenames.append(input_file)
                    train_labels.append(patients_dict[patient_code])
"""
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-T1_brain_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-T1_brain_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                #print(input_file, patients_dict[patient_code])
                train_filenames.append(input_file)
                train_labels.append(patients_dict[patient_code])
            if patient_code in valid_patients:
                valid_filenames.append(input_file)
                valid_labels.append(patients_dict[patient_code])
print("Total Number of valid patients: "+str(len(valid_filenames)))
print("Total Number of train patients: "+str(len(train_filenames)))
#filenames = filenames[1:55]
#labels = labels[1:55]
# Split data into training and validation
train = (train_filenames, train_labels)
validation = (valid_filenames, valid_labels)

train_data = DataInput(params=config.config.get('parameters'), data=train)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation)
# T1 baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)
'''

