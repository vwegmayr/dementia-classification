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

mode = 'DTI_MD'
data_path = '/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'

filep = open('../valid/valid_patients_CBF.pkl', 'rb')
valid_patients = pickle.load(filep)

train_patients = []

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
valid_X_1 = []
valid_Y_1 = []
valid_X_2 = []
valid_Y_2 = []
valid_X_3 = []
valid_Y_3 = []

CBF_filep = open('../features_10/DTI_MD_features_10.pkl', 'rb')
CBF_indices = pickle.load(CBF_filep)

T1_filep = open('../features_10/DTI_FA_features_10.pkl', 'rb')
T1_indices = pickle.load(T1_filep)

DTI_MO_filep = open('../features_10/DTI_MO_features_10.pkl', 'rb')
DTI_MO_indices = pickle.load(DTI_MO_filep)

DTI_folder = '/home/rams/4_Sem/Thesis/Data/DTI_MO_subsampled/'
T1_folder = '/home/rams/4_Sem/Thesis/Data/DTI_FA_subsampled/'

"""
for directory in os.walk('/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'):
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
            if patient_code in train_patients:
                #print(input_file, patients_dict[patient_code])
                #print("Adding", input_file)
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                #print(len(mri_image))
                feature_selected_image = np.take(mri_image, CBF_indices)
                if len(feature_selected_image) > 0:
                    train_X.append(feature_selected_image)
                    train_Y.append(patients_dict[patient_code])
                # Add T1 modality
                T1_filename = T1_folder + patient_code + '/' + patient_code
                T1_filename += '-DTI_FA_subsampled.nii.gz'
                #print("Adding", T1_filename)
                mri_image = nb.load(T1_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                feature_selected_image = np.take(mri_image, T1_indices)
                if len(feature_selected_image) > 0:
                    train_X.append(feature_selected_image)
                    train_Y.append(patients_dict[patient_code])

                # Add DTI MO modality
                DTI_filename = DTI_folder + patient_code
                DTI_filename += '-DTI_MO_subsampled.nii.gz'
                #print("Adding", DTI_filename)
                mri_image = nb.load(DTI_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                # print(len(mri_image))
                feature_selected_image = np.take(mri_image, DTI_MO_indices)
                if len(feature_selected_image) > 0:
                    train_X.append(feature_selected_image)
                    train_Y.append(patients_dict[patient_code])

            if patient_code in valid_patients:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                # print(len(mri_image))
                feature_selected_image = np.take(mri_image, CBF_indices)
                if len(feature_selected_image) > 0:
                    valid_X_1.append(feature_selected_image)
                    valid_Y_1.append(patients_dict[patient_code])
                # Add T1 modality
                T1_filename = T1_folder + patient_code + '/' + patient_code
                T1_filename += '-DTI_FA_subsampled.nii.gz'
                mri_image = nb.load(T1_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                # print(len(mri_image))
                feature_selected_image = np.take(mri_image, T1_indices)
                if len(feature_selected_image) > 0:
                    valid_X_2.append(feature_selected_image)
                    valid_Y_2.append(patients_dict[patient_code])

                # Add DTI MO modality
                DTI_filename = DTI_folder + patient_code
                DTI_filename += '-DTI_MO_subsampled.nii.gz'
                mri_image = nb.load(DTI_filename)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()
                # print(len(mri_image))
                feature_selected_image = np.take(mri_image, DTI_MO_indices)
                if len(feature_selected_image) > 0:
                    valid_X_3.append(feature_selected_image)
                    valid_Y_3.append(patients_dict[patient_code])
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
print("Valid", "DTI_MD:", len(valid_X_1), "DTI_FA:", len(valid_X_2), "DTI_MO:", len(valid_X_3))
print("Train", len(train_X))
mode = 'multitask'
ntrain = np.array(train_X)
ntrain_labels = np.array(train_Y)
#np.save(mode+"_train_10", ntrain)
#np.save(mode+"_train_labels_10", ntrain_labels)
nvalid_1  = np.array(valid_X_1)
nvalid_labels_1 = np.array(valid_Y_1)
nvalid_2  = np.array(valid_X_2)
nvalid_labels_2 = np.array(valid_Y_2)
nvalid_3  = np.array(valid_X_3)
nvalid_labels_3 = np.array(valid_Y_3)
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
#print("Storing the model...")
#joblib.dump(classifier, mode+'_classifier.pkl')
print("Grid scores on development set:")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
# FA - Means = [ 0.53549849  0.53549849  0.53549849]
# FA - std = [ 0.06966128  0.06966128  0.06966128] for C:1, 10, 100
print(means, stds)
pred = classifier.predict(nvalid_1)
correct = 0
for i in range(0, len(pred)):
    if pred[i] == nvalid_labels_1[i]:
        correct += 1

print("Accuracy: Task 1 DTI_MD",(correct/len(pred)))
#  FA Accuracy:  0.58 for C:1, kernel: linear

pred = classifier.predict(nvalid_2)
correct = 0
for i in range(0, len(pred)):
    if pred[i] == nvalid_labels_2[i]:
        correct += 1

print("Accuracy: Task 2 DTI_FA",(correct/len(pred)))

pred = classifier.predict(nvalid_3)
correct = 0
for i in range(0, len(pred)):
    if pred[i] == nvalid_labels_3[i]:
        correct += 1

print("Accuracy: Task 3 DTI_MO",(correct/len(pred)))
