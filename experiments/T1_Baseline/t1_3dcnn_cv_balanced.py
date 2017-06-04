from os import path
import os
import pickle
import re
import argparse
import sys
import subprocess

from dementia_prediction.config_wrapper import Config
from dementia_prediction.cnn_baseline.data_input_balanced import DataInput
from dementia_prediction.cnn_baseline.baseline_balanced import CNN

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
#s_split = int(paths['validation_split']*len(s_codes))
#p_split = int(paths['validation_split']*len(p_codes))
# Divide the data into train, valid and test
#s_codes = s_codes[:1]
#p_codes = p_codes[:1]
for i in range(0,10):
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
    subprocess.call('python experiments/T1_Baseline/t1_3dcnn_cv_balanced_input.py params.yaml {0} '
                    '{1} {2} {3} {4}'.format(str(s_start), str(s_end), str(p_start),
                                                str(p_end), str(i+1)), shell=True)
