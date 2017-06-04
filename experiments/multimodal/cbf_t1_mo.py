from os import path
import os
import pickle
import re
import argparse
import sys

from dementia_prediction.config_wrapper import Config
from dementia_prediction.cnn_baseline.multimodal_input import DataInput
from dementia_prediction.cnn_baseline.multimodal_baseline import CNN

config = Config()
param_file = sys.argv[1]
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
        regex = r"-CBF_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-CBF_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patients_dict[patient_code] == 0:
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    p_codes.append(patient_code)
print("Total Number of Stable Patients: "+str(len(s_codes)))
print("Total Number of Progressive Patients: "+str(len(p_codes)))
#s_split = int(paths['validation_split']*len(s_codes))
#p_split = int(paths['validation_split']*len(p_codes))
# Divide the data into train, valid and test
#s_codes = s_codes[:1]
#p_codes = p_codes[:1]
s_split = 20
p_split = 20
train_patients = s_codes[s_split:]+p_codes[p_split:]
valid_patients = s_codes[:s_split]+p_codes[:p_split]
#train_patients = s_codes+p_codes
#valid_patients = s_codes+p_codes
#print(train_patients)
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
            regex = r"-CBF_sub_rot3_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-CBF_sub_rot3_{0}.nii.gz'
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
            regex = r"-CBF_sub_rot3_trans3_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-CBF_sub_rot3_trans3_{0}.nii.gz'
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
            regex = r"-CBF_sub_trans3_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-CBF_sub_trans3_{0}.nii.gz'
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
        regex = r"-CBF_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-CBF_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict and patient_code in train_patients:
                if patients_dict[patient_code] == 0:
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
#filenames = filenames[1:55]
#labels = labels[1:55]
# Split data into training and validation
train = (pos_train_filenames, neg_train_filenames)
validation = (pos_valid_filenames, neg_valid_filenames)

train_data = DataInput(params=config.config.get('parameters'), data=train, name='train')
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation, name='valid')
# T1 baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)


