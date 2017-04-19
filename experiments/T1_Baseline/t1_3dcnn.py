from os import path
import os
import pickle
import re

from dementia_prediction.config_wrapper import Config
from dementia_prediction.cnn_baseline.data_input import DataInput
from dementia_prediction.cnn_baseline.baseline import CNN

config = Config()
config.parse(path.abspath("params.yaml"))

paths = config.config.get('data_paths')
regex = r"-T1_brain_smoothed\.nii\.gz$"
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)

filenames = []
labels = []

for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-T1_brain_smoothed.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                filenames.append(input_file)
                labels.append(patients_dict[patient_code])
print("Total Number of patients: "+str(len(filenames)))
print("Demented: "+str(sum(labels)))
print("Stable: "+str((len(filenames) - sum(labels))))

# Split data into training and validation
split = int(len(filenames)*paths['validation_split'])
train = (filenames[split:], labels[split:])
validation = (filenames[:split], labels[:split])

print("Train: Demented: "+str(sum(train[1]))+" Stable: "+str((len(
    train[1])-sum(train[1]))))
print("Validation: Demented: "+str(sum(validation[1]))+" Stable: "+str((len(
    validation[1])-sum(validation[1]))))

train_data = DataInput(params=config.config.get('parameters'), data=train)
validation_data = DataInput(params=config.config.get('parameters'),
                            data=validation)
# T1 baseline CNN model
cnn_model = CNN(params=config.config.get('parameters'))
cnn_model.train(train_data, validation_data)



