


import nibabel as nb
import numpy as np
import os
import math
from os import path
import re
import sys
import pickle
from skimage import util
import sklearn.preprocessing as pre
import scipy.stats as sc
from time import gmtime, strftime

class RobustFisher:


    def __init__(self, mode):
        self.mode = mode
        self.min_patient = []
        self.max_patient = []
        self.mean_progressive = []
        self.mean_stable = []
        self.std_progressive = []
        self.std_stable = []

        self.fisher_score = []

        self.prob_progressive = []
        self.prob_stable = []
        self.patch_size = 3
        self.image_size = 897600
        self.varA = 0
        self.bins_patient = []
        self.patients_dict = pickle.load(open(path.abspath('./patients.pkl'),
                                          'rb'))
        print("Hello")
        for i in range(0, self.image_size):
            self.min_patient.append(sys.maxsize)
            self.max_patient.append(-sys.maxsize)

            self.std_progressive.append(0)
            self.std_stable.append(0)

            self.fisher_score.append(0)

            self.prob_progressive.append([])
            self.prob_stable.append([])

    def min_max(self, filenames):
        for input_file in filenames:
            print(input_file)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            sys.stdout.flush()
            # print(input_file, patients_dict[patient_code])
            mri_image = nb.load(input_file)
            mri_image = mri_image.get_data()
            # mri_image_flat = pre.scale(mri_image_flat, copy=False)
            # with open('image.pkl', 'wb') as filep:
            #    pickle.dump(mri_image_flat, filep)

            # mri_image_re = mri_image.reshape(88, 102, 100)
            mri_image_padded = util.pad(mri_image, 1, 'constant')
            winshape = (self.patch_size, self.patch_size, 1)
            windows = util.view_as_windows(mri_image_padded,
                                           window_shape=winshape)
            # max_ = 0
            # min_ = sys.maxsize()
            ctr = 0

            for i in range(0, 88):
                for j in range(0, 102):
                    for k in range(1, 101):
                        arr = np.reshape(windows[i, j, k],
                                         -1).tolist()
                        if np.max(arr) > self.max_patient[ctr]:
                            self.max_patient[ctr] = np.max(arr)
                        if np.min(arr) < self.min_patient[ctr]:
                            self.min_patient[ctr] = np.min(arr)
                        ctr += 1
        return self.min_patient, self.max_patient

    def histogram_dist(self, filenames):
        for input_file in filenames:
            pat_code = input_file.rsplit('-' + self.mode +
                                         '_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            print(input_file)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            sys.stdout.flush()
            # print(input_file, patients_dict[patient_code])
            mri_image = nb.load(input_file)
            mri_image = mri_image.get_data()
            # TODO: normalization
            mri_image_padded = util.pad(mri_image, 1, 'constant')
            winshape = (self.patch_size, self.patch_size, 1)
            windows = util.view_as_windows(mri_image_padded,
                                           window_shape=winshape)
            # print(len(mri_image))
            ctr = 0
            for i in range(0, 88):
                for j in range(0, 102):
                    for k in range(1, 101):
                        if self.patients_dict[patient_code] == 1:
                            if len(self.bins_patient[ctr]) != 1:
                                #print("Bins:", self.bins_patient[ctr])
                                sys.stdout.flush()
                                arr = np.reshape(windows[i, j, k],
                                                 -1).tolist()
                                values, base = np.histogram(arr,
                                                        self.bins_patient[ctr],
                                                        normed=False)
                                values_prob = values / len(arr)
                                if len(self.prob_progressive[ctr]) == 0:
                                    self.prob_progressive[ctr] = values_prob
                                else:
                                    self.prob_progressive[ctr] = [x + y for x, y in zip(
                                        self.prob_progressive[ctr], values_prob)]
                                #if len(self.prob_progressive[ctr]) == 0:
                                #    print(arr, self.bins_patient[ctr])
                                #    sys.stdout.flush()
                                #    sys.exit()
                            else:
                                self.prob_progressive[ctr] = [0]
                                #print("Ctr:", ctr, "Prob: 0")
                                sys.stdout.flush()

                        else:
                            if len(self.bins_patient[ctr]) != 1:
                                #print("Bins:", self.bins_patient[ctr])
                                sys.stdout.flush()
                                arr = np.reshape(windows[i, j, k],
                                                 -1).tolist()
                                values, base = np.histogram(arr,
                                                            self.bins_patient[
                                                                ctr],
                                                            normed=False)
                                values_prob = values / len(arr)
                                if len(self.prob_stable[ctr]) == 0:
                                    self.prob_stable[ctr] = values_prob
                                else:
                                    self.prob_stable[ctr] = [x + y for x, y in zip(
                                        self.prob_stable[ctr], values_prob)]
                            else:
                                self.prob_stable[ctr] = [0]
                                #print("Ctr:", ctr, "Prob: 0")
                                sys.stdout.flush()
                        ctr += 1
        return self.prob_progressive, self.prob_stable

    def std_dev(self, filenames):
        for input_file in filenames:
            pat_code = input_file.rsplit('-' + self.mode +
                                         '_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]

            print(input_file)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            sys.stdout.flush()
            mri_image = nb.load(input_file)
            mri_image = mri_image.get_data()

            mri_image_padded = util.pad(mri_image, 1, 'constant')
            winshape = (self.patch_size, self.patch_size, 1)
            windows = util.view_as_windows(mri_image_padded,
                                           window_shape=winshape)
            ctr = 0
            for i in range(0, 88):
                for j in range(0, 102):
                    for k in range(1, 101):
                        if self.patients_dict[patient_code] == 1:
                            if len(self.bins_patient[ctr]) != 1:
                                arr = np.reshape(windows[i, j, k],
                                             -1).tolist()
                                values, base = np.histogram(arr,
                                                            self.bins_patient[ctr],
                                                            normed=False)
                                if len(arr) == 0 or len(self.mean_progressive[
                                                            ctr]) == 0 or \
                                                len(values) == 0:
                                    print("Yes")
                                values_prob = values / len(arr)
                                self.std_progressive[ctr] += sc.entropy(
                                                                   values_prob,
                                                                   self.mean_progressive[
                                                                       ctr])
                                # print("std prog", std_progressive[i])
                                '''
                                print("Ctr: ", ctr, "arr:", arr, "Values Prob: ",
                                  values_prob,
                                  "Bins: ", self.bins_patient[ctr],
                                  "Mean Prog: ", self.mean_progressive[ctr],
                                  "Std: ",
                                  self.std_progressive[ctr])
                                '''
                            else:
                                self.std_progressive[ctr] = 0
                                #print("Ctr:", ctr, "Std: 0")
                        else:
                            if len(self.bins_patient[ctr]) != 1:

                                arr = np.reshape(windows[i, j, k],
                                                 -1).tolist()
                                values, base = np.histogram(arr,
                                                            self.bins_patient[ctr],
                                                            normed=False)
                                values_prob = values / len(arr)

                                self.std_stable[ctr] += sc.entropy(values_prob,
                                                              self.mean_stable[
                                                                  ctr])
                                '''
                                print("Ctr: ", ctr, "arr:", arr, "Values Prob: ",
                                  values_prob,
                                  "Bins: ", self.bins_patient[ctr],
                                  "Mean Stab: ", self.mean_stable[ctr],
                                  "Std: ",
                                  self.std_stable[ctr])
                                '''
                            else:
                                self.std_stable[ctr] = 0
                                #print("Ctr:", ctr, "Std: 0")
                        ctr += 1
                        # print("std stable", std_stable[i])
        return self.std_progressive, self.std_stable
