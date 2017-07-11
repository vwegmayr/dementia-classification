import os
from os import path
import re
import sys
import random
import numpy as np
from sklearn import svm
import nibabel as nb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

filep = open(path.abspath('../patients.pkl'), 'rb')
patients_dict = pickle.load(filep)

filep = open('../valid/valid_patients_CBF.pkl', 'rb')
valid_patients = pickle.load(filep)

train_patients = []

mode = 'CBF'
data_path = '/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'

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
                if patient_code not in valid_patients:
                    train_patients.append(patient_code)


train_X = []
train_Y = []
valid_X = []
valid_Y = []

"""
s_shuffle_indices = list(range(len(s_codes)))
random.shuffle(s_shuffle_indices)
s_codes = [s_codes[i] for i in s_shuffle_indices]
p_shuffle_indices = list(range(len(p_codes)))
random.shuffle(p_shuffle_indices)
p_codes = [p_codes[i] for i in p_shuffle_indices]
"""

CBF_filep = open('../features_10/CBF_features_10.pkl', 'rb')
CBF_indices = pickle.load(CBF_filep)

filep = open('../fisher_10/CBF_fisher_score_10.pkl', 'rb')
cbf_scores = pickle.load(filep)


T1_filep = open('../features_10/T1_brain_features_10.pkl', 'rb')
T1_indices = pickle.load(T1_filep)

filep = open('../fisher_10/T1_brain_fisher_score_10.pkl', 'rb')
t1_scores = pickle.load(filep)

DTI_FA_filep = open('../features_10/DTI_FA_features_10.pkl', 'rb')
DTI_FA_indices = pickle.load(DTI_FA_filep)

filep = open('../fisher_10/DTI_FA_fisher_score_10.pkl', 'rb')
dti_FA_scores = pickle.load(filep)

cbf_lim = t1_lim = dti_FA_lim = num_features = 0
while num_features < 89760:
    if dti_FA_lim < len(dti_FA_scores):
        if t1_scores[t1_lim] <= dti_FA_scores[dti_FA_lim] and cbf_scores[cbf_lim] <= dti_FA_scores[dti_FA_lim]:
            dti_FA_lim += 1
            num_features += 1
            continue
    if t1_lim < len(t1_scores):
        if t1_scores[t1_lim] >= cbf_scores[cbf_lim] and t1_scores[t1_lim] >= dti_FA_scores[dti_FA_lim]:
            t1_lim += 1
            num_features += 1
            continue
    if cbf_lim < len(cbf_scores):
        if t1_scores[t1_lim] <= cbf_scores[cbf_lim] and dti_FA_scores[dti_FA_lim] <= cbf_scores[cbf_lim]:
            cbf_lim += 1
            num_features += 1
            continue
T1_indices = T1_indices[:t1_lim]
DTI_FA_indices = DTI_FA_indices[:dti_FA_lim]
CBF_indices = CBF_indices[:cbf_lim]
print("T1", t1_lim, "DTI_FA", dti_FA_lim, "CBF", cbf_lim)

DTI_folder = '/home/rams/4_Sem/Thesis/Data/DTI_FA_subsampled/'
T1_folder = '/home/rams/4_Sem/Thesis/Data/T1_brain_subsampled/'

"""
for directory in os.walk('/home/rams/4_Sem/Thesis/Data/'+FAde+'_subsampled/'):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_brain_sub_rot3_trans3_{0}\.nii\.gz$".format(
                rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-T1_brain_sub_rot3_trans3_{'
                                             '0}.nii.gz'
                                             .format(rotation))
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code in train_patients:
                    mri_image = nb.load(input_file)
                    mri_image = mri_image.get_data()
                    mri_image = mri_image.flatten()
                    #print(len(mri_image))
                    feature_selected_image = np.take(mri_image, indices)
                    if len(feature_selected_image) > 0:
                        train_X.append(feature_selected_image)
                        train_Y.append(patients_dict[patient_code])
                    #print(input_file, patients_dict[patient_code])
                    #train_filenames.append(input_file)
                    #train_labels.append(patients_dict[patient_code])
for directory in os.walk('/home/rams/4_Sem/Thesis/Data/NIFTI/'):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        for rotation in ['x', 'y', 'z']:
            regex = r"-T1_brain_sub_rotation5_{0}\.nii\.gz$".format(rotation)
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-T1_brain_sub_rotation5_{'
                                             '0}.nii.gz'
                                             .format(rotation))
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in patients_dict and patient_code in train_patients:
                    mri_image = nb.load(input_file)
                    mri_image = mri_image.get_data()
                    mri_image = mri_image.flatten()
                    #print(len(mri_image))
                    feature_selected_image = np.take(mri_image, indices)
                    if len(feature_selected_image) > 0:
                        train_X.append(feature_selected_image)
                        train_Y.append(patients_dict[patient_code])
                    #print(input_file, patients_dict[patient_code])
                    #train_filenames.append(input_file)
"""

for directory in os.walk(data_path):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-" + mode + "_subsampled\.nii\.gz$"
        split_on = '-' + mode + '_subsampled.nii.gz'
        if re.search(regex, input_file):
            pat_code = input_file.rsplit(split_on)
            patient_code = pat_code[0].rsplit('/', 1)[1]
            feature_selected_image = []
            if patient_code in patients_dict:
                #print(input_file, patients_dict[patient_code])
                #print("Adding", input_file)
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                #print(len(mri_image))
                feature_selected_image = np.take(mri_image, CBF_indices)
                # Add T1 modality
                T1_filename = T1_folder + patient_code + '/' + patient_code
                T1_filename += '-T1_brain_subsampled.nii.gz'
                #print("Adding", T1_filename)
                mri_image = nb.load(T1_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                feature_selected_image = np.append(feature_selected_image, 
                                           np.take(mri_image, T1_indices))

                # Add DTI MO modality
                DTI_filename = DTI_folder + patient_code + '/' + patient_code
                DTI_filename += '-DTI_FA_subsampled.nii.gz'
                #print("Adding", DTI_filename)
                mri_image = nb.load(DTI_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                # print(len(mri_image))
                feature_selected_image = np.append(feature_selected_image, 
                                           np.take(mri_image, DTI_FA_indices))
            if patient_code in train_patients and len(feature_selected_image) > 0:
                train_X.append(feature_selected_image)
                train_Y.append(patients_dict[patient_code])
            if patient_code in valid_patients and len(feature_selected_image) > 0:
                valid_X.append(feature_selected_image)
                valid_Y.append(patients_dict[patient_code])
'''
# Dont use
train_len = int(0.85*len(X))
ntrain = np.array(X[:train_len])
ntrain_labels = np.array(Y[:train_len])
print("Train:", len(ntrain))
nvalid = np.array(X[train_len:])
nvalid_labels = np.array(Y[train_len:])
print("Test:", len(nvalid))
print("Progressive:", np.sum(nvalid_labels))
'''
print("Valid", len(valid_X))
print("Train", len(train_X))
mode = 'multimodal'
ntrain = np.array(train_X)
ntrain_labels = np.array(train_Y)
#np.save(mode+"_train_10", ntrain)
#np.save(mode+"_train_labels_10", ntrain_labels)
nvalid  = np.array(valid_X)
nvalid_labels = np.array(valid_Y)
#np.save(mode+"_valid_10", nvalid)
#np.save(mode+"_valid_labels_10", nvalid_labels)
"""
ntrain = np.load("train_10.npy")
ntrain_labels = np.load("train_labels_10.npy")
nvalid = np.load("valid_10.npy")
nvalid_labels = np.load("valid_labels_10.npy")
print("Train:", len(ntrain))
print("Valid:", len(nvalid))
"""
# Training grid search
tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]
print("Training..")
sys.stdout.flush()
#classifier = svm.SVC(C=1) #acc 50%
#classifier = svm.SVC(C=10, kernel='linear') #acc 64% with 50 validation set T1
classifier = GridSearchCV(svm.SVC(C=10), tuned_parameters, cv=10, n_jobs=3,
                          scoring='accuracy')
classifier.fit(ntrain,ntrain_labels)
print("Best parameters set found is:")
print(classifier.best_params_) #{'C': 1, 'kernel': 'linear'} for FA
print("Storing the model...")
#joblib.dump(classifier, mode+'_classifier.pkl')
print("Grid scores on development set:")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
# FA - Means = [ 0.53549849  0.53549849  0.53549849]
# FA - std = [ 0.06966128  0.06966128  0.06966128] for C:1, 10, 100
print(means, stds)
pred = classifier.predict(nvalid)
correct = 0
for i in range(0, len(pred)):
    if pred[i] == nvalid_labels[i]:
        correct += 1

print("Accuracy: ",(correct/len(pred)))
#  FA Accuracy:  0.58 for C:1, kernel: linear
