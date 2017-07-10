from os import path
import os
import pickle
import re
import argparse
import sys
import nibabel as nb
import subprocess
import math
import numpy as np

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config

IMG_SIZE = 897600
config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))



paths = config.config.get('data_paths')
filep = open(paths['class_labels'], 'rb')
patients_dict = pickle.load(filep)
filep = open(paths['valid_data'], 'rb')
valid_patients = pickle.load(filep)
print(len(valid_patients))

class Normalize():

    def __init__(self, mode):
        self.mode = mode
        self.mean_path = paths['norm_mean_var'] + \
                         'multimodal/'+self.mode+'_train_data_mean_path.pkl'
        self.var_path = paths['norm_mean_var'] + \
                        'multimodal/'+self.mode+'_train_data_var_path.pkl'

    def mean_fun(self, filenames):
        mean = [0 for x in range(0, IMG_SIZE)]
        for file in filenames:
            image = nb.load(file).get_data().flatten()
            for i in range(0, IMG_SIZE):
                mean[i] += image[i]
        return mean

    def var_fun(self, filenames):
        variance = [0 for i in range(0, IMG_SIZE)]
        with open(self.mean_path, 'rb') as filep:
            global_mean = pickle.load(filep)
        for file in filenames:
            image = nb.load(file).get_data().flatten()
            for i in range(0, IMG_SIZE):
                variance[i] += math.pow((image[i] - global_mean[i]), 2)
        return variance

    def normalize(self, train_data):
        mean = [0 for i in range(0, IMG_SIZE)]
        var = [0 for i in range(0, IMG_SIZE)]
        num_parallel = 15
        split = int(len(train_data) / num_parallel)
        pool = ProcessPool(num_parallel)
        train_splits = []
        for par in range(0, num_parallel - 1):
            train_splits.append(train_data[par * split:(par + 1) * split])
        train_splits.append(train_data[(num_parallel - 1) * split:])

        if not os.path.exists(self.mean_path):
            mean_arrays = pool.map(self.mean_fun, train_splits)
            for i in range(0, IMG_SIZE):
                for j in range(0, len(mean_arrays)):
                    mean[i] += mean_arrays[j][i]
            global_mean = [i/len(train_data) for i in mean]
            with open(self.mean_path, 'wb') as filep:
                pickle.dump(global_mean, filep)
        with open(self.mean_path, 'rb') as filep:
            global_mean = pickle.load(filep)
        if not os.path.exists(self.var_path):
            var_arrays = pool.map(self.var_fun, train_splits)
            for i in range(0, IMG_SIZE):
                for j in range(0, len(var_arrays)):
                    var[i] += var_arrays[j][i]
            global_variance = [math.sqrt(x/(len(train_data)-1)) for x in var]
            with open(self.var_path, 'wb') as filep:
                pickle.dump(global_variance, filep)
        with open(self.var_path, 'rb') as filep:
            global_variance = pickle.load(filep)

        return global_mean, global_variance

train_patients = []

for directory in os.walk(paths['CBF_path']):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-CBF_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.\
                        rsplit('-CBF_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patient_code not in valid_patients:
                    train_patients.append(patient_code)

print("Train: ", len(train_patients))
for mode in ['T1_brain', 'DTI_FA']:
    train = []
    valid = []
    norm_object = Normalize(mode)

    print("Mode:", mode)
    for directory in os.walk(paths[mode+'_path']):
        # Walk inside the directory
        for file in directory[2]:
            # Match all files ending with 'regex'
            input_file = os.path.join(directory[0], file)
            regex = r"-"+mode+"_subsampled\.nii\.gz$"
            if re.search(regex, input_file):
                pat_code = input_file. \
                    rsplit('-'+mode+'_subsampled.nii.gz')
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict:
                    # CBF train images as reference
                    if patient_code in train_patients:
                        train.append(input_file)
                    elif patient_code in valid_patients:
                        valid.append(input_file)
    print("Train: ", len(train), "Valid:", len(valid))
    total = train+valid
    print("Total:", len(total))
    mean_norm, var_norm = norm_object.normalize(train)
    print("Normalized")
    for filename in total:
        mri_image = nb.load(filename)
        affine = mri_image.get_affine()
        mri_image = mri_image.get_data().flatten()
        norm_image = []
        for x, y, z in zip(mri_image, mean_norm, var_norm):
            if z == 0:
                norm_image.append(0)
            else:
                norm_image.append((x - y) / z)
        reshaped_image = np.reshape(norm_image, [88, 102, 100])
        im = nb.Nifti1Image(reshaped_image, affine=affine)
        pat_code = filename.rsplit('-' + mode + '_subsampled.nii.gz')
        patient_code = pat_code[0].rsplit('/', 1)[1]
        output = paths['multimodal_norm_out']+'train/'+mode+'/'+patient_code\
                 +'-'+mode\
                 +'_norm_subsampled.nii.gz'
        print("Saving to output: ", output)
        nb.save(im, output) 
