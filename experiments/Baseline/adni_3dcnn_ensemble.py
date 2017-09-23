from os import path
import os
import pickle
import re
import argparse
import sys
import nibabel as nb
import subprocess
import math

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config
from dementia_prediction.data_input import DataInput
from dementia_prediction.cnn_baseline.ensemble_adni import CNNEnsemble

# Parse the parameter file
config = Config()
parser = argparse.ArgumentParser(description="Run the ADNI Ensemble model")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
paths = config.config.get('data_paths')

patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
train_patients = pickle.load(open(paths['train_data'], 'rb'))
valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
print("Valid Patients:", len(valid_patients), "Train:", len(train_patients))
cad_patients = pickle.load(open(paths['cad_dict'], 'rb'))
print("CAD:", len(cad_patients))

classes = params['cnn']['classes']
train_filenames = [[] for i in range(0, classes)]
valid_filenames = [[] for i in range(0, classes)]
cad_filenames = [[] for i in range(0, classes)]

for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"" + params['regex'] + "$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                train_filenames[patients_dict[patient_code]].append(
                    input_file)
            if patient_code in valid_patients:
                valid_filenames[patients_dict[patient_code]].append(
                    input_file)

for i in range(0, classes):
    print("Train Class ", i, len(train_filenames[i]))
    print("Valid Class ", i, len(valid_filenames[i]))

train_data = DataInput(params=config.config.get('parameters'),
                       data=train_filenames,
                       name='train', mean=0, var=0)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=valid_filenames, name='valid', mean=0,
                            var=0)

for directory in os.walk(paths['caddir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"" + params['regex'] + "$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in cad_patients:
                cad_filenames[cad_patients[patient_code]].append(
                    input_file)

for i in range(0, classes):
    print("CAD Class ", i, len(cad_filenames[i]))

cad_data = DataInput(params=config.config.get('parameters'),
                       data=cad_filenames,
                       name='train', mean=0, var=0)

cnn_model = CNNEnsemble(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, cad_data, True)


