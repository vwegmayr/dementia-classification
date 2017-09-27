from os import path
import os
import pickle
import re
import argparse
import sys
import subprocess

from dementia_prediction.config_wrapper import Config

config = Config()
parser = argparse.ArgumentParser(description="Generate Cross Validation dictionaries")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('data_paths')


patients_dict = pickle.load(open(params['class_labels'], 'rb'))
print("Patients", len(patients_dict))

s_codes = []
p_codes = []

for directory in os.walk(params['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r""+params['regex']+"$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patients_dict[patient_code] == 0:
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    p_codes.append(patient_code)
print("Total Number of Stable Patients: "+str(len(s_codes)))
print("Total Number of Progressive Patients: "+str(len(p_codes)))
for i in range(0, 10):
    s_start = i*12
    s_end = s_start + 12
    total = len(s_codes)
    if s_end > total:
        s_start = s_start - (s_end - total)
        s_end = s_start + 12
    p_start = i*12
    p_end = p_start + 12
    total = len(p_codes)
    if p_end > total:
        p_start = p_start - (p_end - total)
        p_end = p_start + 12
    print("S start", s_start, "S end", s_end, "P Start", p_start, "P end", p_end)
    train_patients = s_codes[:s_start]+s_codes[s_end:]+p_codes[:p_start]+p_codes[p_end:]
    valid_patients = s_codes[s_start:s_end]+p_codes[p_start:p_end]
    print("Train: ", len(train_patients))
    print("Valid: ", len(valid_patients))
    with open(params['dictionary_path']+'OASIS_train_'+str(i+1)+'.pkl', 'wb') as filep:
        pickle.dump(train_patients, filep)    
    with open(params['dictionary_path']+'OASIS_valid_'+str(i+1)+'.pkl', 'wb') as filep:
        pickle.dump(valid_patients, filep)    
