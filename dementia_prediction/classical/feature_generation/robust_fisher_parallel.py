
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
from pathos.multiprocessing import ProcessPool
from classical.robust_fisher_class import RobustFisher

#np.seterr(all='print')

mode = 'DTI_MD'
#data_path = '/local/home/saitharaju/NIFTI/'
data_path = '/local/home/saitharaju/NIFTI/'+mode+'_subsampled/'

NUM_BINS = 3 
IMG_SIZE = 897600

filep = open(path.abspath('../patients.pkl'), 'rb')
patients_dict = pickle.load(filep)


min_patient = []
max_patient = []

mean_progressive = []
mean_stable = []

std_progressive = []
std_stable = []

fisher_score = []

bins_patient = []

prob_progressive = []
prob_stable = []

for i in range(0, IMG_SIZE):

    min_patient.append(sys.maxsize)
    max_patient.append(-sys.maxsize)


    mean_progressive.append([])
    std_progressive.append(0)
    mean_stable.append([])
    std_stable.append(0)

    fisher_score.append(0)

    bins_patient.append([])
    bins_patient.append([])

    prob_progressive.append([])
    prob_stable.append([])


prog_class_ctr = 0
stab_class_ctr = 0
patch_size = 3

# Take CBF patients as reference for valid patients
filep = open('./valid_patients_CBF.pkl', 'rb')
valid_patients_list = pickle.load(filep)

valid_patients = []
train_patients = []
train_filenames = []

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
                if patient_code in valid_patients_list:
                    valid_patients.append(patient_code)
                else:
                    train_patients.append(patient_code)
                    train_filenames.append(input_file)
                    if patients_dict[patient_code] == 1:
                        prog_class_ctr += 1
                    if patients_dict[patient_code] == 0:
                        stab_class_ctr += 1
print("Train: ", len(train_patients), "Valid: ", len(valid_patients))
print("Prog:", prog_class_ctr, "Stab:", stab_class_ctr)
print("Finding min and max intensities for each voxel in a class..")
sys.stdout.flush()
num_parallel = 8 
split = int(len(train_patients)/num_parallel)
robust = RobustFisher(mode)
pool = ProcessPool(num_parallel)
train_splits = []
for par in range(0, num_parallel-1):
    train_splits.append(train_filenames[par*split:(par+1)*split])
train_splits.append(train_filenames[(num_parallel-1)*split:])
print(train_splits)
result_min_max = pool.map(robust.min_max, train_splits)

print("Combining parallel results..")
sys.stdout.flush()
for i in range(0, IMG_SIZE):
    min_list = []
    max_list = []
    for j in range(0, len(result_min_max)):
        min_list.append(result_min_max[j][0][i])
        max_list.append(result_min_max[j][1][i])
    min_patient[i] = np.min(min_list)
    max_patient[i] = np.max(max_list)

with open('./'+mode+'_min_patient_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(min_patient, p_filep)
with open('./' + mode + '_max_patient_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(max_patient, p_filep)
'''
with open('./' + mode + '_min_patient_parallel_newbins.pkl', 'rb') as p_filep:
    min_patient = pickle.load(p_filep)
with open('./' + mode + '_max_patient_parallel_newbins.pkl', 'rb') as p_filep:
    max_patient = pickle.load(p_filep)
'''
print("Deciding the bin boundaries..")
sys.stdout.flush()
#create bin arrays with Min max array for each voxel.
for i in range(0, IMG_SIZE):
    # Put bin boundaries
    if min_patient[i] ==  0 or min_patient[i] == max_patient[i]:
        bins_patient[i] = [0]
    else:
        bins_patient[i] = np.linspace(min_patient[i],
                                      max_patient[i], (NUM_BINS+1))
        print("Ctr:", i, "Bins:", bins_patient[i], "min: ", min_patient[i], "Max:", max_patient[i])

with open('./'+mode+'_bin_patient_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(bins_patient, p_filep)


print("Finding histogram prob. distributions for each voxel..")
sys.stdout.flush()
# Traverse again update for each window i,j,k histogram arrays of size
# 88*102*100
robust.bins_patient = bins_patient
result_hist = pool.map(robust.histogram_dist, train_splits)

for i in range(0, IMG_SIZE):
    for j in range(0, len(result_hist)):
        if len(result_hist[j][0][i]) != 0:
            if len(prob_progressive[i]) == 0:
                prob_progressive[i] = result_hist[j][0][i]
            elif result_hist[j][0][i] == [0]:
                prob_progressive[i] = [0]
            else:
                prob_progressive[i] = [x + y for x, y in zip(
                    prob_progressive[i], result_hist[j][0][i])]
        if len(result_hist[j][1][i]) != 0:
            if len(prob_stable[i]) == 0:
                prob_stable[i] = result_hist[j][1][i]
            elif result_hist[j][1][i] == [0]:
                prob_stable[i] = [0]
            else:
                prob_stable[i] = [x + y for x, y in zip(
                    prob_stable[i], result_hist[j][1][i])]


print("Finding mean distribution of voxels..")
min_len_prog = sys.maxsize
min_len_stab = sys.maxsize
for i in range(0, IMG_SIZE):
    mean_progressive[i] = [x/prog_class_ctr for x in prob_progressive[i]]
    mean_stable[i] = [x/stab_class_ctr for x in prob_stable[i]]
    if len(mean_progressive[i]) < min_len_prog:
        min_len_prog = len(mean_progressive[i])
    if len(mean_stable[i]) < min_len_stab:
        min_len_stab = len(mean_stable[i])
print("Min lengths prog and stable", min_len_prog, min_len_stab)
sys.stdout.flush()

with open('./'+mode+'_mean_prog_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(mean_progressive, p_filep)
with open('./' + mode + '_mean_stab_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(mean_stable, p_filep)
print("Progressive:", len(mean_progressive), "Ctr: ", prog_class_ctr)
print("Stable:", len(mean_stable), "Ctr: ", stab_class_ctr)


print("Finding std. dev. of voxels..")
sys.stdout.flush()
'''
with open('./'+mode+'_bin_patient_parallel_newbins.pkl', 'rb') as p_filep:
    bins_patient = pickle.load(p_filep)
with open('./'+mode+'_mean_prog_parallel_newbins.pkl', 'rb') as p_filep:
    mean_progressive = pickle.load(p_filep)
with open('./' + mode + '_mean_stab_parallel_newbins.pkl', 'rb') as p_filep:
    mean_stable = pickle.load(p_filep)

robust.bins_patient = bins_patient
'''
robust.mean_progressive = mean_progressive
robust.mean_stable = mean_stable

result_std_dev = pool.map(robust.std_dev, train_splits)

for i in range(0, IMG_SIZE):
    for j in range(0, len(result_std_dev)):
        std_progressive[i] += result_std_dev[j][0][i]
        std_stable[i] += result_std_dev[j][1][i]

std_progressive = [x/prog_class_ctr for x in std_progressive]
std_stable = [x/stab_class_ctr for x in std_stable]

with open('./'+mode+'_std_prog_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(std_progressive, p_filep)
with open('./' + mode + '_std_stab_parallel_newbins.pkl', 'wb') as p_filep:
    pickle.dump(std_stable, p_filep)
'''
with open('./'+mode+'_mean_prog_parallel_newbins.pkl', 'rb') as p_filep:
    mean_progressive = pickle.load(p_filep)
with open('./' + mode + '_mean_stab_parallel_newbins.pkl', 'rb') as p_filep:
    mean_stable = pickle.load(p_filep)
with open('./'+mode+'_std_prog_parallel_newbins.pkl', 'rb') as p_filep:
    std_progressive = pickle.load(p_filep)
with open('./' + mode + '_std_stab_parallel_newbins.pkl', 'rb') as p_filep:
    std_stable = pickle.load(p_filep)
with open('./'+mode+'_bin_patient_parallel_newbins.pkl', 'rb') as p_filep:
    bins_patient = pickle.load(p_filep)
with open('./' + mode + '_min_patient_parallel_newbins.pkl', 'rb') as p_filep:
    min_patient = pickle.load(p_filep)
with open('./' + mode + '_max_patient_parallel_newbins.pkl', 'rb') as p_filep:
    max_patient = pickle.load(p_filep)
#print("Std Progressive", std_progressive)
#print("Std Stable", std_stable)
'''
print("Processing Fisher scores..")
sys.stdout.flush()

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
for i in range(0, IMG_SIZE):
    numerator = 0
    denominator = 0
    fisher_score[i] = 0
    if len(mean_progressive[i]) != 1 and len(mean_stable[i]) != 1:
        numerator = sc.entropy(mean_progressive[i], mean_stable[i]) +\
                          sc.entropy(mean_stable[i], mean_progressive[i])
    denominator = std_progressive[i] + std_stable[i]

    if np.isinf(denominator):
        print("i:", i, " Num:", numerator, " Den: inf")
        sys.exit()
    elif denominator != 0:
        fisher_score[i] = numerator/denominator

    print("i:", i, "Min:", min_patient[i], "\nMax:", max_patient[i],
          "\nBins:", bins_patient[i], "\nMeanProg:",
          mean_progressive[i],"\nMeanStab:",
          mean_stable[i],"\nNum:",
          numerator,"\nDen:", denominator, fisher_score[i])
    sys.stdout.flush()
#print("Fisher score",fisher_score[i])
print("Sorting fisher scores..")

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
# sort the indices
sys.stdout.flush()
with open('./fisher_score_10/'+mode+'_fisher_score_robust_parallel_all_newbins.pkl', 'wb') as \
        p_filep:
    pickle.dump(fisher_score, p_filep)
sorted_fisher_indices = sorted(range(len(fisher_score)),
                           key=lambda k: fisher_score[k], reverse=True)
# Take top 10%
significant_len = int(0.1*len(fisher_score))
final_indices = sorted_fisher_indices[:significant_len]
fisher_score_10 =  [fisher_score[x] for x in final_indices]
with open('./features_10/'+mode+'_features_10_robust_parallel_newbins.pkl', 'wb') as \
        p_filep:
    pickle.dump(final_indices, p_filep)
with open('./fisher_score_10/'+mode+'_fisher_score_10_robust_parallel_newbins.pkl',
          'wb') as \
        p_filep:
    pickle.dump(fisher_score_10, p_filep)

# Take top 15%
significant_len = int(0.15*len(fisher_score))
final_indices = sorted_fisher_indices[:significant_len]
fisher_score_15 =  [fisher_score[x] for x in final_indices]
with open('./features_15/'+mode+'_features_15_robust_parallel_newbins.pkl', 'wb') as \
        p_filep:
    pickle.dump(final_indices, p_filep)
with open('./fisher_score_15/'+mode+'_fisher_score_15_robust_parallel_newbins.pkl', 'wb') as \
        p_filep:
    pickle.dump(fisher_score_15, p_filep)
