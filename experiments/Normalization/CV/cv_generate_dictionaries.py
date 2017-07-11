from os import path
import os
import pickle
import re
import argparse
import sys
import subprocess

from dementia_prediction.config_wrapper import Config

config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))


paths = config.config.get('data_paths')
#regex = r"-T1_brain_sub_rotation5\.nii\.gz$"
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)
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
mode = 'T1_brain'
# Get CBF all filenames
# Divide them into 8 sets of validation and train data
# Create ductionaries of trainand validation data
# Normalize them and store them
for directory in os.walk(paths['CBF_path']):
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
for i in range(0, 9):
    s_start = i*20
    s_end = s_start + 20
    total = len(s_codes)
    if s_end > total:
        s_start = s_start - (s_end - total)
        s_end = s_start + 20
    p_start = i*20
    p_end = p_start + 20
    total = len(p_codes)
    if p_end > total:
        p_start = p_start - (p_end - total)
        p_end = p_start + 20
    print("S start", s_start, "S end", s_end, "P Start", p_start, "P end", p_end)
    train_patients = s_codes[:s_start]+s_codes[s_end:]+p_codes[:p_start]+p_codes[p_end:]
    valid_patients = s_codes[s_start:s_end]+p_codes[p_start:p_end]
    print("Train: ", len(train_patients))
    print("Valid: ", len(valid_patients))
    with open(paths['dictionary_path']+'cv/cv_train_codes_'+str(i+1)+'.pkl', 'wb') as filep:
        pickle.dump(train_patients, filep)    
    with open(paths['dictionary_path']+'cv/cv_valid_codes_'+str(i+1)+'.pkl', 'wb') as filep:
        pickle.dump(valid_patients, filep)    
