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
from dementia_prediction.transfer_learning.data_input_new import DataInput
#TODO: Uncomment this
#from dementia_prediction.cnn_baseline.adni_data_input import DataInput
from dementia_prediction.cnn_baseline.adni_baseline import CNN

IMG_SIZE = 902629 
config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))


paths = config.config.get('data_paths')
filep = open(paths['train_data'], 'rb')
train_patients = pickle.load(filep)
filep = open(paths['valid_data'], 'rb')
valid_patients = pickle.load(filep)
print(len(valid_patients), len(train_patients))

global_mean = [0 for i in range(0, IMG_SIZE)]
global_variance = [0 for i in range(0, IMG_SIZE)]
#mean_path = paths['norm_mean_var']+'./t1_train_mean_path.pkl'
#var_path = paths['norm_mean_var']+'./t1_train_var_path.pkl'

def mean_fun(filenames):
    mean = [0 for x in range(0, IMG_SIZE)]
    for file in filenames:
        image = nb.load(file).get_data().flatten()
        for i in range(0, IMG_SIZE):
            mean[i] += image[i]
    return mean
def var_fun(filenames):
    variance = [0 for i in range(0, IMG_SIZE)]
    with open(mean_path, 'rb') as filep:
        global_mean = pickle.load(filep)
    for file in filenames:
        image = nb.load(file).get_data().flatten()
        for i in range(0, IMG_SIZE):
            variance[i] += math.pow((image[i] - global_mean[i]), 2)#TODO check globalmean
    return variance
def normalize(train):
    train_data = train[0] + train[1]
    mean = [0 for i in range(0, IMG_SIZE)]
    var = [0 for i in range(0, IMG_SIZE)]
    num_parallel = 15
    split = int(len(train_data) / num_parallel)
    pool = ProcessPool(num_parallel)
    train_splits = []
    for par in range(0, num_parallel - 1):
        train_splits.append(train_data[par * split:(par + 1) * split])
    train_splits.append(train_data[(num_parallel - 1) * split:])
    '''
    # If argument list too long, skip this
    if not os.path.exists(mean_path):
        command = 'fslmaths'
        command += ' -add '.join(train_data)
        command += ' -div ' + str(len(train_data)) + \
                   ' ' + mean_path
        print("Number of training images: " + str(len(train_data)))
        subprocess.call(command, shell=True)
    mean = nb.load(mean_path).get_data()
    if not os.path.exists(mean_path):
        mean = [0 for x in range(IMG_SIZE)]
        for file in train_data:
            print("mean", file)
            mr_image = nb.load(file).get_data().flatten()
            for i in range(0, IMG_SIZE):
                mean[i] += mr_image[i]
        mean = [i/len(train_data) for i in mean]
        with open(mean_path, 'wb') as filep:
            pickle.dump(mean, filep)
    if not os.path.exists(var_path):
        variance = [0 for x in range(IMG_SIZE)]
        #mean_values = nb.load(mean_path).get_data().flatten()
        with open(mean_path, 'rb') as filep:
            mean_values = pickle.load(filep)
        for file in train_data:
            print("var", file)
            image = nb.load(file).get_data().flatten()
            for i in range(0, IMG_SIZE):
                variance[i] += math.pow((image[i] - mean_values[i]), 2)
        variance = [math.sqrt(x/(len(train_data)-1)) for x in variance]
        with open(var_path, 'wb') as filep:
            pickle.dump(variance, filep)
    '''
    if not os.path.exists(mean_path):
        mean_arrays = pool.map(mean_fun, train_splits)
        for i in range(0, IMG_SIZE):
            for j in range(0, len(mean_arrays)):
                mean[i] += mean_arrays[j][i]
        global_mean = [i/len(train_data) for i in mean]
        with open(mean_path, 'wb') as filep:
            pickle.dump(global_mean, filep)
    with open(mean_path, 'rb') as filep:
        global_mean = pickle.load(filep)
    if not os.path.exists(var_path):
        var_arrays = pool.map(var_fun, train_splits)
        for i in range(0, IMG_SIZE):
            for j in range(0, len(var_arrays)):
                var[i] += var_arrays[j][i]
        global_variance = [math.sqrt(x/(len(train_data)-1)) for x in var]
        with open(var_path, 'wb') as filep:
            pickle.dump(global_variance, filep)
    with open(var_path, 'rb') as filep:
        global_variance = pickle.load(filep)

    return global_mean, global_variance

nc_train_filenames = []
mci_train_filenames = []
ad_train_filenames = []
nc_valid_filenames = []
mci_valid_filenames = []
ad_valid_filenames = []
"""
# Add Data Augmented with rotations and translations
for directory in os.walk(paths['datadir']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_translated_{0}_affine_corrected\.nii\.gz$".format(rotation)
            split_on = ''
            if re.search(regex, input_file):
                split_on = '-T1_translated_{0}_affine_corrected.nii.gz'.format(rotation)
            regex = r"-T1_brain_sub_rotation5_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                split_on = '-T1_brain_sub_rotation5_{0}.nii.gz'.format(rotation)
            '''
            regex = r"-T1_brain_subsampled\.nii\.gz_sub_rot3_trans3_{0}\.nii\.gz$".format(
                rotation)
            if re.search(regex, input_file):
                split_on = '-T1_brain_subsampled.nii.gz_sub_rot3_trans3_{0}.nii.gz'.format(rotation)
            '''
            if split_on != '':
                pat_code = input_file.rsplit(split_on)
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in train_patients and patient_code not in \
                        valid_patients:
                    if train_patients[patient_code] == 0:
                        s_train_filenames.append(input_file)
                    if train_patients[patient_code] == 1:
                        p_train_filenames.append(input_file)
print(len(s_train_filenames), len(p_train_filenames))
# 166*6 stable and 180*6 progressive patients
"""
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


