from os import path
import os
import pickle
import re
import argparse
import sys
import subprocess

from dementia_prediction.config_wrapper import Config

config = Config()

parser = argparse.ArgumentParser(description="Run the Cross Validation "
                                             "baseline")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
paths = config.config.get('data_paths')
params = config.config.get('parameters')

patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
s_codes = []
p_codes = []

for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = params['regex']
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
    subprocess.call('python '
                    'experiments/Baseline/3dcnn_cv_input.py '
                    'experiments/Baseline/parameters/'+params['cnn'][
                    'mode']+'_cv_params.yaml {0} '
                    '{1} {2} {3} {4}'.format(str(s_start), str(s_end), str(p_start),
                                                str(p_end), str(i+1)), shell=True)
