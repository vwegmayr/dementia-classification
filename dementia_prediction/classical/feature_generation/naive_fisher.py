
import nibabel as nb
import numpy as np
import os
import math
from os import path
import re
import sys
import pickle

# Patient code and his class label.
filep = open(path.abspath('./patients.pkl'), 'rb')
patients_dict = pickle.load(filep)

mode = 'DTI_MO'
data_path = '/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'

mean_progressive = []
mean_stable = []
std_progressive = []
std_stable = []
fisher_score = []

IMG_SIZE = 88*102*100 #897600

for i in range(0, IMG_SIZE):
    mean_progressive.append(0)
    std_progressive.append(0)
    mean_stable.append(0)
    std_stable.append(0)
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
        regex = r"-"+mode+"_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-'+mode+'_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()

                if patients_dict[patient_code] == 1:
                    prog_class_ctr += 1
                if patients_dict[patient_code] == 0:
                    stab_class_ctr += 1
                for i in range(0,len(mri_image)):
                    if patients_dict[patient_code] == 1:
                        mean_progressive[i] += mri_image[i]
                    else:
                        mean_stable[i] += mri_image[i]
mean_progressive = [x/prog_class_ctr for x in mean_progressive]
mean_stable = [x/stab_class_ctr for x in mean_stable]

# Inner loop takes an image and updates the standard dev. of each voxel
for directory in os.walk(data_path):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-"+mode+"_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-'+mode+'_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()

                for i in range(0,len(mri_image)):
                    if patients_dict[patient_code] == 1:
                        std_progressive[i] += math.pow((mri_image[i]
                                                        - mean_progressive[i]),
                                                       2)

                    else:
                        std_stable[i] += math.pow((mri_image[
                                                            i] -
                                                        mean_stable[i]),
                                                       2)
std_progressive = [math.sqrt(x/prog_class_ctr) for x in std_progressive]
std_stable = [math.sqrt(x/stab_class_ctr) for x in std_stable]

# Fisher score
for i in range(0, len(fisher_score)):
    fisher_score[i] = math.pow((mean_progressive[i] - mean_stable[i]), 2)

    if (std_progressive[i] + std_stable[i]) == 0:
        fisher_score[i] = 0
    else:
        fisher_score[i] = fisher_score[i]/(std_progressive[i] + std_stable[i])


# sort the indices
sorted_fisher_indices = sorted(range(len(fisher_score)),
                               key=lambda k: fisher_score[k], reverse=True)
# Take top 10%
significant_len = int(0.1*len(fisher_score))
final_indices = sorted_fisher_indices[:significant_len]
fisher_score_10 =  [fisher_score[x] for x in final_indices]
with open(mode+'_features_10_new.pkl', 'wb') as p_filep:
    pickle.dump(final_indices, p_filep)
with open(mode+'_fisher_score_10_new.pkl', 'wb') as p_filep:
    pickle.dump(fisher_score_10, p_filep)

# Take top 15%
significant_len = int(0.15*len(fisher_score))
final_indices = sorted_fisher_indices[:significant_len]
fisher_score_15 =  [fisher_score[x] for x in final_indices]
with open(mode+'_features_15_new.pkl', 'wb') as p_filep:
    pickle.dump(final_indices, p_filep)
with open(mode+'_fisher_score_15_new.pkl', 'wb') as p_filep:
    pickle.dump(fisher_score_15, p_filep)
