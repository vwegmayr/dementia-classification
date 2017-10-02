
import nibabel as nb
import numpy as np
import os
import math
from os import path
import re
import sys
import pickle
import sklearn.preprocessing as pre
import argparse

from pathos.multiprocessing import ProcessPool
from dementia_prediction.config_wrapper import Config

config = Config()
parser = argparse.ArgumentParser(description="Generate General Fisher Score "
                                             "Features")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')
data_path = params['datadir']

# Patient code and his class label.
patients_dict = pickle.load(open(params['patient_dict'], 'rb'))
train_patients = pickle.load(open(params['train_dict'], 'rb'))
output_features = params['features_path']
print("Patients:", len(patients_dict), "Train: ", len(train_patients))

IMG_SIZE = params['depth']*params['height']*params['width']


mean_progressive = []
mean_stable = []
mean_total = []
var_progressive = []
var_stable = []
fisher_score = []
for i in range(0, IMG_SIZE):
    mean_progressive.append(0)
    var_progressive.append(0)
    mean_stable.append(0)
    var_stable.append(0)
    mean_total.append(0)
    fisher_score.append(0)


# Finding mean of each pixel for a class
prog_class_ctr = 0
stab_class_ctr = 0

# Inner loop takes an image and updates the mean values for all voxels
for directory in os.walk(data_path):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r""+params['regex']+"$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                #mri_image = pre.scale(mri_image, copy=False)

                if patients_dict[patient_code] == 1:
                    prog_class_ctr += 1
                if patients_dict[patient_code] == 0:
                    stab_class_ctr += 1
                for i in range(0,len(mri_image)):
                    if patients_dict[patient_code] == 1:
                        mean_progressive[i] += mri_image[i]
                    else:
                        mean_stable[i] += mri_image[i]
                    mean_total[i] += mri_image[i]

mean_progressive = [x/prog_class_ctr for x in mean_progressive]
mean_stable = [x/stab_class_ctr for x in mean_stable]
mean_total = [x/(prog_class_ctr+stab_class_ctr) for x in mean_total]

print("Prog class ctr:", prog_class_ctr, "Stable: ", stab_class_ctr)

# Inner loop takes an image and updates the standard dev. of each voxel
for directory in os.walk(data_path):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r""+params['regex']+"$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(params['split_on'])
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                #mri_image = pre.scale(mri_image, copy=False)

                for i in range(0,len(mri_image)):
                    if patients_dict[patient_code] == 1:
                        var_progressive[i] += math.pow((mri_image[i]
                                                        - mean_progressive[i]),
                                                       2)

                    else:
                        var_stable[i] += math.pow((mri_image[
                                                            i] -
                                                        mean_stable[i]),
                                                       2)
var_progressive = [x/prog_class_ctr for x in var_progressive]
var_stable = [x/stab_class_ctr for x in var_stable]

score_zero = 0
# Fisher score
for i in range(0, len(fisher_score)):
    numerator = (prog_class_ctr * math.pow((mean_progressive[i]
                                            - mean_total[i]), 2))+ \
                (stab_class_ctr * math.pow((mean_stable[i]
                                            - mean_total[i]), 2))
    denominator = prog_class_ctr * var_progressive[i] +\
                  stab_class_ctr * var_stable[i]
    if denominator == 0:
        fisher_score[i] = 0
        score_zero += 1
    else:
        fisher_score[i] = numerator/denominator
    print(fisher_score[i], flush=True)
print("There are ", score_zero, "pixels with zero score")
with open(output_features+'fisher_score_all.pkl', 'wb') as p_filep:
    pickle.dump(fisher_score, p_filep)

