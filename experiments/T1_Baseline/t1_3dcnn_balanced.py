from os import path
import os
import pickle
import re
import argparse
import sys
import nibabel as nb
import subprocess
import math

from dementia_prediction.config_wrapper import Config
from dementia_prediction.cnn_baseline.data_input_balanced import DataInput
from dementia_prediction.cnn_baseline.t1_baseline import CNN

IMG_SIZE = 897600
config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))

filep = open('../validation_patients.pkl', 'rb')
valid_patients = pickle.load(filep)

paths = config.config.get('data_paths')
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)

def normalize(train):
    train_data = train[0] + train[1]
    mean_image = './train_mean_image.nii.gz'
    var_image = './train_var_image.pkl'
    mean = []
    var = []

    if not os.path.exists(mean_image):
        command = 'fslmaths'
        command += ' -add '.join(train_data)
        command += ' -div ' + str(len(train_data)) + \
                   ' ' + mean_image
        print("Number of training images: " + str(len(train_data)))
        subprocess.call(command, shell=True)

    if not os.path.exists(var_image):
        variance = [0 for x in range(IMG_SIZE)]
        mean_values = nb.load(mean_image).get_data()
        for file in train_data:
            image = nb.load(file).get_data()
            for i in range(0, 897600):
                variance[i] += math.pow((image[i] - mean_values[i]), 2)
        variance = [x/len(train_data) for x in variance]
        with open(var_image, 'wb') as filep:
            pickle.dump(variance, filep)

    mean = nb.load(mean_image).get_data()
    with open(var_image, 'rb') as filep:
        var = pickle.load(filep)

    return mean, var

s_train_filenames = []
p_train_filenames = []
s_valid_filenames = []
p_valid_filenames = []

# Add Data Augmented with rotations and translations
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_translated_{0}_affine_corrected\.nii\.gz$".format(rotation)
            split_on = ''
            if re.search(regex, input_file):
                split_on = '-T1_translated_{0}_affine_corrected.nii.gz'.format(rotation)
            regex = r"-T1_brain_sub_rotation5_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                split_on = '-T1_brain_sub_rotation5_{0}.nii.gz'.format(rotation)
            '''
            regex = r"-T1_brain_subsampled\.nii\.gz_sub_rot3_trans3_{0}\.nii\.gz$".format(
                rotation)
            if re.search(regex, input_file):
                split_on = '-T1_brain_subsampled.nii.gz_sub_rot3_trans3_{0}.nii.gz'.format(rotation)
            '''
            if split_on != '':
                pat_code = input_file.rsplit(split_on)
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code not in \
                        valid_patients:
                    if patients_dict[patient_code] == 0:
                        s_train_filenames.append(input_file)
                    if patients_dict[patient_code] == 1:
                        p_train_filenames.append(input_file)

for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-T1_brain_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-T1_brain_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict and patient_code not in \
                    valid_patients:
                if patients_dict[patient_code] == 0:
                    s_train_filenames.append(input_file)
                if patients_dict[patient_code] == 1:
                    p_train_filenames.append(input_file)
            if patient_code in patients_dict and patient_code in valid_patients:
                if patients_dict[patient_code] == 0:
                    s_valid_filenames.append(input_file)
                if patients_dict[patient_code] == 1:
                    p_valid_filenames.append(input_file)

print("Train Data: S: ", len(s_train_filenames), "P: ", len(p_train_filenames))
print("Validation Data: S: ", len(s_valid_filenames), "P: ", len(p_valid_filenames))

train = (s_train_filenames, p_train_filenames)
validation = (s_valid_filenames, p_valid_filenames)
mean_norm, var_norm = normalize(train)
train_data = DataInput(params=config.config.get('parameters'), data=train,
                       name='train', mean=mean_norm, var=var_norm)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation, name='valid', mean=mean_norm,
                            var=var_norm)
# T1 baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)


