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
from dementia_prediction.fusion_input import FusionDataInput
from dementia_prediction.multichannel_input import MultichannelDataInput

from dementia_prediction.data_input_perceptron import DataInputPerceptron
from dementia_prediction.cnn_baseline.baseline_model import CNNBaseline
from dementia_prediction.multimodal.finetuning_conv7 import CNNMultimodal
from dementia_prediction.multimodal.ensemble import EnsembleCNN

# Parse the parameter file
config = Config()
parser = argparse.ArgumentParser(description="Run the Baseline model")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
paths = config.config.get('data_paths')

# All patients class labels dictionary and list of validation patient codes
patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
train_patients = pickle.load(open(paths['train_data'], 'rb'))
print("Validation patients count in Dict: ", len(valid_patients),
      "Train patients count in Dict:", len(train_patients))

classes = params['cnn']['classes']
train_filenames = [[] for i in range(0, classes)]
valid_filenames = [[] for i in range(0, classes)]

multitask = ['']
# If True, multitask takes all the modalities as input to the same network
if params['multitask'] == 'True':
    multitask = range(0, 3)
for modality in multitask:
    for directory in os.walk(paths['datadir'+str(modality)]):
        # Walk inside the directory
        for file in directory[2]:
            # Match all files ending with 'regex'
            input_file = os.path.join(directory[0], file)
            regex = r""+params['regex']+"$"
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

'''
# For augmentation of data on the fly
aug_train = [[] for i in range(0, classes)]
for i in range(0, classes):
    for filename in train_filenames[i]:
        aug_train[i].append(filename)
        for direction in ['x','y','z']:
            aug_train[i].append(filename+'rot'+direction)
            aug_train[i].append(filename+'trans'+direction)
train_filenames = aug_train
print("After augmentation:")
for i in range(0, classes):
    print("Train Class ", i, len(train_filenames[i]))
    print("Valid Class ", i, len(valid_filenames[i]))
'''

train_data = DataInput(params=config.config.get('parameters'),
                       data=train_filenames,
                       name='train', mean=0, var=0)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=valid_filenames, name='valid', mean=0,
                            var=0)

cnn_model = CNNBaseline(params=config.config.get('parameters'))

if params['mlp'] == 'True':
    train_data = DataInputPerceptron(params=config.config.get('parameters'),
                                     data=train_filenames,
                       name='train', mean=0, var=0)
    validation_data = DataInputPerceptron(params=config.config.get('parameters'),
                            data=valid_filenames, name='valid', mean=0,
                            var=0)

if params['cnn']['channels'] > 1:
    train_data = MultichannelDataInput(params=config.config.get('parameters'),
                                     data=train_filenames,
                       name='train', mean=0, var=0)
    validation_data = MultichannelDataInput(params=config.config.get('parameters'),
                            data=valid_filenames, name='valid', mean=0,
                            var=0)
if params['multimodal'] == 'True':
    train_data = FusionDataInput(params=config.config.get('parameters'),
                                 data=train_filenames,
                                 name='train', mean=0, var=0)
    validation_data = FusionDataInput(params=config.config.get('parameters'),
                                      data=valid_filenames, name='valid',
                                      mean=0,
                                      var=0)
    cnn_model = CNNMultimodal(params=config.config.get('parameters'))
if params['multimodal'] == 'ensemble':
    train_data = FusionDataInput(params=config.config.get('parameters'),
                                 data=train_filenames,
                                 name='train', mean=0, var=0)
    validation_data = FusionDataInput(params=config.config.get('parameters'),
                                      data=valid_filenames, name='valid',
                                      mean=0,
                                      var=0)
    cnn_model = EnsembleCNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data, True)

