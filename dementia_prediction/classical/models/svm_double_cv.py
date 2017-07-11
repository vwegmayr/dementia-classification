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

filep = open(path.abspath('./patients.pkl'), 'rb')
patients_dict = pickle.load(filep)
mode = 'DTI_MO'
filep = open('./'+mode+'_features_10_new.pkl', 'rb')
indices = pickle.load(filep)
print("Using mode: "+mode)
in_filenames = []
train_labels = []
valid_filenames = []
valid_labels = []
s_codes = []
p_codes = []

for directory in os.walk('/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        #regex = r"-DTI_FA_subsampled\.nii\.gz$"
        regex = r"-"+mode+"_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-'+mode+'_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in patients_dict:
                if patients_dict[patient_code] == 0:
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    p_codes.append(patient_code)
print("Stable:", len(s_codes), "Progressive:", len(p_codes))

run = 1
for i in range(0,10):
    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    s_start = i*20
    s_end = s_start + 20
    total = len(s_codes)
    if s_end > total:
        s_start = s_start - (s_end - total)
        s_end = s_start + 20
    p_start = i*20
    p_end = p_start + 20
    total = len(p_codes)
    if p_end > total:
        p_start = p_start - (p_end - total)
        p_end = p_start + 20
    train_patients = s_codes[:s_start]+s_codes[s_end:]+p_codes[:p_start]+p_codes[p_end:]
    valid_patients = s_codes[s_start:s_end]+p_codes[p_start:p_end]
    print("Train: ", train_patients)
    print("Valid: ", valid_patients)

    for directory in os.walk('/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'):
        # Walk inside the directory
        for file in directory[2]:
            # Match all files ending with 'regex'
            input_file = os.path.join(directory[0], file)
            regex = r"-"+mode+"_subsampled\.nii\.gz$"
            if re.search(regex, input_file):
                pat_code = input_file.rsplit('-'+mode+'_subsampled.nii.gz')
                patient_code = pat_code[0].rsplit('/', 1)[1]
                if patient_code in train_patients:
                    #print(input_file, patients_dict[patient_code])
                    mri_image = nb.load(input_file)
                    mri_image = mri_image.get_data()
                    mri_image = mri_image.flatten()
                    #print(len(mri_image))
                    feature_selected_image = np.take(mri_image, indices)
                    if len(feature_selected_image) > 0:
                        train_X.append(feature_selected_image)
                        train_Y.append(patients_dict[patient_code])
                if patient_code in valid_patients:
                    #print(input_file, patients_dict[patient_code])
                    mri_image = nb.load(input_file)
                    mri_image = mri_image.get_data()
                    mri_image = mri_image.flatten()
                    #print(len(mri_image))
                    feature_selected_image = np.take(mri_image, indices)
                    if len(feature_selected_image) > 0:
                        valid_X.append(feature_selected_image)
                        valid_Y.append(patients_dict[patient_code])

    print("Train:", len(train_X))
    print("Valid:", len(valid_X))
    ntrain = np.array(train_X)
    ntrain_labels = np.array(train_Y)
    np.save(mode+"_train_aug_10", ntrain)
    np.save(mode+"_train_labels_aug_10", ntrain_labels)
    nvalid  = np.array(valid_X)
    nvalid_labels = np.array(valid_Y)
    np.save(mode+"_valid_aug_10", nvalid)
    np.save(mode+"_valid_labels_aug_10", nvalid_labels)
    """
    ntrain = np.load("train_10.npy")
    ntrain_labels = np.load("train_labels_10.npy")
    nvalid = np.load("valid_10.npy")
    nvalid_labels = np.load("valid_labels_10.npy")
    print("Train:", len(ntrain))
    print("Valid:", len(nvalid))
    """
    # Training grid search
    tuned_parameters = [{'kernel': ['linear', 'rbf'], 'C': [1]}]
    print("Training..")
    sys.stdout.flush()
    #classifier = svm.SVC(C=1) #acc 50%
    #classifier = svm.SVC(C=10, kernel='linear') #acc 64% with 50 validation set T1
    classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, n_jobs=3,
                              scoring='accuracy')
    classifier.fit(ntrain,ntrain_labels)
    print("Best parameters set found is:")
    print(classifier.best_params_) #{'C': 1, 'kernel': 'linear'} for FA
    print("Storing the model...")
    joblib.dump(classifier, mode+'_classifier.pkl')
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

    print("Run ", run, "Accuracy: ",(correct/len(pred)))

    sys.stdout.flush()
    run += 1
    #  FA Accuracy:  0.58 for C:1, kernel: linear
