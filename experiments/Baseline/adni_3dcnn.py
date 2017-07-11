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
from dementia_prediction.data_input_new import DataInput
from dementia_prediction.cnn_baseline.adni_baseline import CNN

# Parse the parameter file
config = Config()
parser = argparse.ArgumentParser(description="Run the ADNI Baseline model")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
paths = config.config.get('data_paths')

IMG_SIZE = params['cnn']['depth']*params['cnn']['height']*params['cnn'][
            'width']

paths = config.config.get('data_paths')
train_patients = pickle.load(open(paths['train_data'], 'rb'))
valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
print(len(valid_patients), len(train_patients))


nc_train_filenames = []
mci_train_filenames = []
ad_train_filenames = []
nc_valid_filenames = []
mci_valid_filenames = []
ad_valid_filenames = []

print(valid_patients)
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"_normalized\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('_normalized.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            #print(patient_code)
            if patient_code in train_patients:
                if train_patients[patient_code] == 0:
                    nc_train_filenames.append(input_file)
                if train_patients[patient_code] == 1:
                    mci_train_filenames.append(input_file)
                if train_patients[patient_code] == 2:
                    ad_train_filenames.append(input_file)
            if patient_code in valid_patients:
                if valid_patients[patient_code] == 0:
                    nc_valid_filenames.append(input_file)
                if valid_patients[patient_code] == 1 or\
                   valid_patients[patient_code] == 4:
                    mci_valid_filenames.append(input_file)
                if valid_patients[patient_code] == 2 or\
                   valid_patients[patient_code] == 5:
                    ad_valid_filenames.append(input_file)

print("Train Data: NC: ", len(nc_train_filenames), "MCI: ", len(mci_train_filenames), "AD:", len(ad_train_filenames))
print("Validation Data: NC: ", len(nc_valid_filenames), "MCI: ", len(mci_valid_filenames), "AD:", len(ad_valid_filenames), 
        "Total:", len(nc_valid_filenames)+len(mci_valid_filenames)+len(ad_valid_filenames))

train = (nc_train_filenames, mci_train_filenames, ad_train_filenames)
validation = (nc_valid_filenames, mci_valid_filenames, ad_valid_filenames)
train_data = DataInput(params=config.config.get('parameters'), data=train,
                       name='train', mean=0, var=0)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation, name='valid', mean=0,
                            var=0)
# ADNI baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)


