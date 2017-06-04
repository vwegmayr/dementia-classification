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

mode = 'CBF'
data_path = '/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'

modalities = {'CBF' : 0, 'DTI_MO' : 1, 'DTI_MD' : 2, 'DTI_FA' : 3,
              'T1_brain' : 4 }

filep = open('../valid/valid_patients_CBF.pkl', 'rb')
valid_patients = pickle.load(filep)
print("Multi task 10% features")

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

valid_X = [[] for i in range(0, len(modalities))]
valid_Y = [[] for i in range(0, len(modalities))]
features = [[] for i in range(0, len(modalities))]
folder_path = {}
for (key, value) in modalities.items():
    features[value] = pickle.load(open('../features/mask/robust/' + key +
                                       '_features_10.0_robust_mask.pkl', 'rb'))
    folder_path[value] = '/home/rams/4_Sem/Thesis/Data/'+key+'_subsampled/'

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
            if patient_code in patients_dict:
                for (key, value) in modalities.items():
                    if key == 'T1_brain' or key == 'DTI_FA':
                        input_file = folder_path[value] + patient_code + '/' + \
                                      patient_code + '-' + key + \
                                     '_subsampled.nii.gz'
                    else:
                        input_file = folder_path[value] + patient_code + '-' +\
                                     key + '_subsampled.nii.gz'
                    mri_image = nb.load(input_file)
                    mri_image = mri_image.get_data()
                    mri_image = mri_image.flatten()
                    feature_selected_image = np.take(mri_image,
                                                     features[value])
                    if patient_code in train_patients:
                        train_X.append(feature_selected_image)
                        train_Y.append(patients_dict[patient_code])
                    elif patient_code in valid_patients:
                        valid_X[value].append(feature_selected_image)
                        valid_Y[value].append(patients_dict[patient_code])

mode = 'multitask'
ntrain = np.array(train_X)
ntrain_labels = np.array(train_Y)
print("Train: ", len(train_X))
"""
np.save(mode+"_train_10", ntrain)
np.save(mode+"_train_labels_10", ntrain_labels)
np.save(mode+"_valid_10", nvalid)
np.save(mode+"_valid_labels_10", nvalid_labels)
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

for key, value in modalities.items():
    print("Modality:", key, len(valid_X[value]))
    pred = classifier.predict(np.array(valid_X[value]))
    correct = 0
    for i in range(0, len(pred)):
        if pred[i] == valid_Y[value][i]:
            correct += 1

    print("Accuracy: Task "+key,(correct/len(pred)))
