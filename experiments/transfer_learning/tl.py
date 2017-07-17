from os import path
import os
import pickle
import re
import argparse

from dementia_prediction.config_wrapper import Config
from dementia_prediction.data_input import DataInput
from dementia_prediction.transfer_learning.adni_toptuning import CNN
from dementia_prediction.transfer_learning.adni_finetuning import FinetuneCNN

# Parse the parameter file
config = Config()
parser = argparse.ArgumentParser(description="Run the Transfer Learning model")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
paths = config.config.get('data_paths')

IMG_SIZE = params['cnn']['depth']*params['cnn']['height']*params['cnn'][
            'width']
patients_dict = pickle.load(open(paths['class_labels'], 'rb'))
valid_patients = pickle.load(open(paths['valid_data'], 'rb'))
print(len(valid_patients))
train_patients = pickle.load(open(paths['train_data'], 'rb'))
print(len(train_patients))

s_train_filenames = []
p_train_filenames = []
s_valid_filenames = []
p_valid_filenames = []

for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r""+paths['regex']+"$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(paths['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                if patients_dict[patient_code] == 0:
                    s_train_filenames.append(input_file)
                if patients_dict[patient_code] == 1:
                    p_train_filenames.append(input_file)
            if patient_code in valid_patients:
                if patients_dict[patient_code] == 0:
                    s_valid_filenames.append(input_file)
                if patients_dict[patient_code] == 1:
                    p_valid_filenames.append(input_file)

print("Train Data: S: ", len(s_train_filenames), "P: ", len(p_train_filenames))
print("Validation Data: S: ", len(s_valid_filenames), "P: ", len(p_valid_filenames))

train = (s_train_filenames, p_train_filenames)
validation = (s_valid_filenames, p_valid_filenames)

train_data = DataInput(params=params, data=train,
                       name='train', mean=0, var=0)
validation_data = DataInput(params=params,
                            data=validation, name='valid', mean=0,
                            var=0)
cnn_model = CNN(params=params)
if params['tl'] == 'finetune':
    print("Finetuning...")
    cnn_model = FinetuneCNN(params=params)
cnn_model.train(train_data, validation_data, True)
