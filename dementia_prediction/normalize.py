from os import path
import os
import pickle
import re
import sys
import nibabel as nb
import math
import numpy as np
from pathos.multiprocessing import ProcessPool


class Normalize():

    def __init__(self, params, train_dict, valid_dict):
        self.IMG_SIZE = params['img_size']
        self.params = params
        self.mode = params['mode']
        self.mean_path = params['norm_mean_var'] + self.mode + '_data_mean.pkl'
        self.var_path = params['norm_mean_var'] + self.mode + '_data_var.pkl'
        self.mean_norm = []
        self.var_norm = []
        self.train_dict = train_dict
        self.valid_dict = valid_dict

    def get_files(self, dir, regex, split_on):
        train_patients = []
        valid_patients = []
        for directory in os.walk(dir):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                if re.search(regex, input_file):
                    pat_code = input_file.rsplit(split_on)
                    patient_code = pat_code[0].rsplit('/', 1)[1]
                    if patient_code in self.train_dict:
                        train_patients.append(input_file)
                    elif patient_code in self.valid_dict:
                        valid_patients.append(input_file)
        return train_patients, valid_patients

    def mean_fun(self, filenames):
        mean = [0 for x in range(0, self.params['img_size'])]
        for file in filenames:
            image = nb.load(file).get_data().flatten()
            for i in range(0, self.params['img_size']):
                mean[i] += image[i]
        return mean

    def var_fun(self, filenames):
        variance = [0 for i in range(0, self.params['img_size'])]
        with open(self.mean_path, 'rb') as filep:
            global_mean = pickle.load(filep)
        for file in filenames:
            image = nb.load(file).get_data().flatten()
            for i in range(0, self.params['img_size']):
                variance[i] += math.pow((image[i] - global_mean[i]), 2)
        return variance

    def per_image(self, train_data):
        for filename in train_data:
            mri_image = nb.load(filename)
            affine = mri_image.get_affine()
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            mri_image = (mri_image - mean) / (stddev + self.params['epsilon'])
            reshaped_image = np.reshape(mri_image, [self.params['height'],
                                                    self.params['width'],
                                                    self.params['depth']])
            im = nb.Nifti1Image(reshaped_image, affine=affine)
            patient = filename.rsplit('/', 1)[1]
            output = self.params['per_image_out'] + patient
            print("Saving to output: ", output)
            nb.save(im, output)

    def store(self, data):
        for filename in data:
            pat_code = filename.rsplit(self.params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            output = self.params['norm_out'] + patient_code + \
                                           '_normalized.nii.gz'
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
                        norm_image.append((x - y) / (z + self.params['epsilon']))
                mean = np.array(norm_image).mean()
                std = np.array(norm_image).std()
                norm_image = (norm_image - mean) / (std + self.params['epsilon'])
                reshaped_image = np.reshape(norm_image, [self.params['height'],
                                                    self.params['width'],
                                                    self.params['depth']])
                im = nb.Nifti1Image(reshaped_image, affine=affine)
                print("Saving to output: ", output)
                nb.save(im, output)

    def normalize(self, train_data):
        mean = [0 for i in range(0, self.params['img_size'])]
        var = [0 for i in range(0, self.params['img_size'])]
        num_parallel = 20
        split = int(len(train_data) / num_parallel)
        pool = ProcessPool(num_parallel)
        train_splits = []
        for par in range(0, num_parallel - 1):
            train_splits.append(train_data[par * split:(par + 1) * split])
        train_splits.append(train_data[(num_parallel - 1) * split:])

        if not os.path.exists(self.mean_path):
            mean_arrays = pool.map(self.mean_fun, train_splits)
            for i in range(0, self.params['img_size']):
                for j in range(0, len(mean_arrays)):
                    mean[i] += mean_arrays[j][i]
            global_mean = [i / len(train_data) for i in mean]
            with open(self.mean_path, 'wb') as filep:
                pickle.dump(global_mean, filep)
            # If you want to view mean image
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
            for i in range(0, self.params['img_size']):
                for j in range(0, len(var_arrays)):
                    var[i] += var_arrays[j][i]
            global_variance = [math.sqrt(x / (len(train_data) - 1)) for x in
                               var]
            with open(self.var_path, 'wb') as filep:
                pickle.dump(global_variance, filep)
        with open(self.var_path, 'rb') as filep:
            global_variance = pickle.load(filep)

        return global_mean, global_variance

