from os import path
import os
import pickle
import re
import argparse
import sys

from dementia_prediction.config_wrapper import Config
from dementia_prediction.data_input_new import DataInput
from dementia_prediction.cnn_baseline.baseline_balanced import CNN

config = Config()
param_file = sys.argv[1]
s_start = int(sys.argv[2])
s_end = int(sys.argv[3])
p_start = int(sys.argv[4])
p_end = int(sys.argv[5])
run = int(sys.argv[6])

config.parse(path.abspath(param_file))


paths = config.config.get('data_paths')
#regex = r"-T1_brain_sub_rotation5\.nii\.gz$"
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)
print(patients_dict)
pos_train_filenames = []
pos_train_labels = []
pos_valid_filenames = []
pos_valid_labels = []
neg_train_filenames = []
neg_train_labels = []
neg_valid_filenames = []
neg_valid_labels = []
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
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patients_dict[patient_code] == 0:
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    p_codes.append(patient_code)
print("Total Number of Stable Patients: "+str(len(s_codes)))
print("Total Number of Progressive Patients: "+str(len(p_codes)))

train_patients = s_codes[:s_start]+s_codes[s_end:]+p_codes[:p_start]+p_codes[p_end:]
valid_patients = s_codes[s_start:s_end]+p_codes[p_start:p_end]
print("Run "+str(run))
print("Train:" + str(train_patients))
print("Valid:" + str(valid_patients))
print("Total Number of unique Train Patients: "+str(len(train_patients)))
print("Total Number of unique Valid Patients: "+str(len(valid_patients)))

"""
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-T1_translated\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-T1_translated.nii.gz')
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
            regex = r"-T1_translated_{0}_affine_corrected\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-T1_translated_{0}_affine_corrected.nii.gz'
                                             .format(rotation))
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code in train_patients:
                    #print(input_file, patients_dict[patient_code])
                    if patients_dict[patient_code] == 0:
                        pos_train_filenames.append(input_file)
                        pos_train_labels.append(patients_dict[patient_code])
                    if patients_dict[patient_code] == 1:
                        neg_train_filenames.append(input_file)
                        neg_train_labels.append(patients_dict[patient_code])
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
                    if patients_dict[patient_code] == 0:
                        pos_train_filenames.append(input_file)
                        pos_train_labels.append(patients_dict[patient_code])
                    if patients_dict[patient_code] == 1:
                        neg_train_filenames.append(input_file)
                        neg_train_labels.append(patients_dict[patient_code])
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
            if patient_code in patients_dict and patient_code in train_patients:
                if patients_dict[patient_code] == 0:#pos is healthy, neg is progressive
                    pos_train_filenames.append(input_file)
                    pos_train_labels.append(patients_dict[patient_code])
                if patients_dict[patient_code] == 1:
                    neg_train_filenames.append(input_file)
                    neg_train_labels.append(patients_dict[patient_code])
            if patient_code in patients_dict and patient_code in valid_patients:
                if patients_dict[patient_code] == 0:
                    pos_valid_filenames.append(input_file)
                    pos_valid_labels.append(patients_dict[patient_code])
                if patients_dict[patient_code] == 1:
                    neg_valid_filenames.append(input_file)
                    neg_valid_labels.append(patients_dict[patient_code])
print("Total Number of valid patients: "+str(len(neg_valid_filenames)+len(pos_valid_filenames)))
print("Total Number of train patients: "+str(len(neg_train_filenames)+len(pos_train_filenames)))
print("Valid pos patients: "+str(pos_valid_filenames))
print("Valid neg patients: "+str(neg_valid_filenames))

# Split data into training and validation
train = (pos_train_filenames, neg_train_filenames)
validation = (pos_valid_filenames, neg_valid_filenames)

train_data = DataInput(params=config.config.get('parameters'), data=train, name='train')
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation, name='valid')
# T1 baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)


