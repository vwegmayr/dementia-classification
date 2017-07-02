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
from collections import defaultdict

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config

IMG_SIZE = 902629 
config = Config()
param_file = sys.argv[1]
config.parse(path.abspath(param_file))
EPSILON = 0.00001


paths = config.config.get('data_paths')

filep = open(paths['adni_valid_path'], 'rb')
valid_dict = pickle.load(filep)
print("Valid", len(valid_dict), "Patients")

filep = open(paths['adni_train_path'], 'rb')
train_dict = pickle.load(filep)
print("Train", len(train_dict), "Patients")

class Normalize():

    def __init__(self, mode):
        self.mode = mode
        self.mean_path = paths['norm_mean_var'] + \
                         'adni/'+self.mode+'_train_data_mean.pkl'
        self.var_path = paths['norm_mean_var'] + \
                        'adni/'+self.mode+'_train_data_var.pkl'
        self.mean_norm = []
        self.var_norm = []
    
    def get_files(self, dir):
        train_patients = []
        valid_patients = []
        for directory in os.walk(dir):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                regex = r"_mni_aligned\.nii\.gz$"
                if re.search(regex, input_file):
                    pat_code = input_file.\
                                rsplit('_mni_aligned.nii.gz')
                    patient_code = pat_code[0].rsplit('/', 1)[1]
                    #print(patient_code, ignore_patients)
                    if patient_code in train_dict:
                        train_patients.append(input_file)
                    elif patient_code in valid_dict:
                        valid_patients.append(input_file)
        return train_patients, valid_patients

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
    
    def per_image(self, train_data):
        for filename in train_data:
            mri_image = nb.load(filename)
            affine = mri_image.get_affine()
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            mri_image = (mri_image - mean) / (stddev+EPSILON)
            reshaped_image = np.reshape(mri_image, [91, 109, 91])
            im = nb.Nifti1Image(reshaped_image, affine=affine)
            patient = filename.rsplit('/', 1)[1]
            output = paths['adni_per_image_out']+patient
            print("Saving to output: ", output)
            nb.save(im, output)
    
    def store(self, data):
        for filename in data:
                pat_code = filename.rsplit('_mni_aligned.nii.gz')
                patient_code = pat_code[0].rsplit('/', 1)[1]
                output = paths['adni_norm_out']+patient_code+'_normalized.nii.gz'
                print("Normalizing image", filename)
                if not os.path.exists(output):
                    mri_image = nb.load(filename)
                    affine = mri_image.get_affine()
                    mri_image = mri_image.get_data().flatten()
                    norm_image = []
                    for x, y, z in zip(mri_image, self.mean_norm, self.var_norm):
                        if z == 0:
                            norm_image.append(0)
                        else:
                            norm_image.append((x - y) / (z+EPSILON))
                    mean = np.array(norm_image).mean()
                    std = np.array(norm_image).std()
                    norm_image = (norm_image - mean) / (std + EPSILON)
                    reshaped_image = np.reshape(norm_image, [91, 109, 91])
                    im = nb.Nifti1Image(reshaped_image, affine=affine)
                    print("Saving to output: ", output)
                    nb.save(im, output)

    def normalize(self, train_data):
        mean = [0 for i in range(0, IMG_SIZE)]
        var = [0 for i in range(0, IMG_SIZE)]
        num_parallel = 20 
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
            '''
            reshaped_image = np.reshape(global_mean, [91, 109, 91])
            sample_affine = nb.load(train_data[0]).get_affine()
            im = nb.Nifti1Image(reshaped_image, affine=sample_affine)
            nb.save(im, './mean_image.nii.gz')
            '''
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


def main():
    norm_object = Normalize('adni')
    train_patients, valid_patients = norm_object.get_files(paths['smooth_path'])
    print("Train: ", len(train_patients), "Valid:", len(valid_patients))
    num_parallel = 20 
    pool = ProcessPool(num_parallel)
    all_patients = train_patients + valid_patients # per-image normalization of all images
    split = int(len(all_patients) / num_parallel)
    splits = []
    for par in range(0, num_parallel - 1):
        splits.append(all_patients[par * split:(par + 1) * split])
    splits.append(all_patients[(num_parallel - 1) * split:])
    print("Finding per-image normalization..")
    pool.map(norm_object.per_image, splits)
    print("Retrieving files from ", paths['adni_per_image_out'])
    train_patients, valid_patients = norm_object.get_files(paths['adni_per_image_out'])
    all_patients = train_patients + valid_patients
    
    print("Finding mean, var normalization of ", len(train_patients), "train images")
    norm_object.mean_norm, norm_object.var_norm = norm_object.normalize(train_patients)
    
    # For normalization on entire dataset
    #print("Finding mean, var normalization of ", len(all_patients), "images")
    #norm_object.mean_norm, norm_object.var_norm = norm_object.normalize(all_patients)
    
    print("Found mean and var")
    split = int(len(all_patients) / num_parallel)
    splits = []
    for par in range(0, num_parallel - 1):
        splits.append(all_patients[par * split:(par + 1) * split])
    splits.append(all_patients[(num_parallel - 1) * split:])
    pool.map(norm_object.store, splits)

if __name__ == '__main__':
    main()
