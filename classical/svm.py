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

data_path = '/home/rams/4_Sem/Thesis/Data/'+mode+'_subsampled/'

train_labels = []
valid_filenames = []
valid_labels = []
s_codes = []
p_codes = []

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
                if patients_dict[patient_code] == 0:
                    # Stable patient
                    s_codes.append(patient_code)
                elif patients_dict[patient_code] == 1:
                    # Progressive patient
                    p_codes.append(patient_code)

print("Stable:", len(s_codes), "Progressive:", len(p_codes))

train_X = []
train_Y = []
valid_X = []
valid_Y = []

# Balanced validation data of 40 patients
s_split = 20
p_split = 20
# Split the data into training and validation sets
train_patients = s_codes[s_split:]+p_codes[p_split:]
valid_patients = s_codes[:s_split]+p_codes[:p_split]

print("Train:", train_patients, "Valid:", valid_patients)

# Add training data to train_X and validation to valid_X
for directory in os.walk(data_path):
    # Walk inside the directory
    for file in directory[2]:
        # Match all files ending with 'regex'
        input_file = os.path.join(directory[0], file)
        regex = r"-"+mode+"_subsampled\.nii\.gz$"
        if re.search(regex, input_file):
            pat_code = input_file.rsplit('-'+mode+'_subsampled.nii.gz')
            patient_code = pat_code[0].rsplit('/', 1)[1]
            if patient_code in train_patients:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()

                feature_selected_image = np.take(mri_image, indices)
                if len(feature_selected_image) > 0:
                    train_X.append(feature_selected_image)
                    train_Y.append(patients_dict[patient_code])

            if patient_code in valid_patients:
                mri_image = nb.load(input_file)
                mri_image = mri_image.get_data()
                mri_image = mri_image.flatten()

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

# Training grid search parameters
tuned_parameters = [{'kernel': ['linear'], 'C': [1]}]

classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, n_jobs=3,
                          scoring='accuracy')
classifier.fit(ntrain,ntrain_labels)

print("Best parameters set found: ", classifier.best_params_)
print("Storing the model...")
joblib.dump(classifier, mode+'_classifier.pkl')

print("Grid scores on development set:")
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
print("Mean: ", means, "Std:", stds)

# Predict on validation data
pred = classifier.predict(nvalid)
correct = 0
for i in range(0, len(pred)):
    if pred[i] == nvalid_labels[i]:
        correct += 1

print("Accuracy: ",(correct/len(pred)))