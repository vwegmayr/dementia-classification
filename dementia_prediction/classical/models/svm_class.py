import os
from os import path
import re
import sys
import random
import numpy as np
from sklearn import svm
import nibabel as nb
import pickle

import sklearn.preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib



class SVM:

    def __init__(self, data_path, mode, valid_path, patients_path):
        self.data_path = data_path + mode +'_subsampled'
        self.mode = mode
        self.patients_dict = pickle.load(open(patients_path, 'rb'))
        self.valid_patients = pickle.load(open(valid_path, 'rb'))
        self.train_patients = []

    def train_and_predict(self, scaling, features):
        """
        Args:
            scaling: Boolean argument to normalize the input image 
            features: features to be selected from the input image.

        Returns: SVM classifier
        """

        # Pick the train data patient names from the data directory
        for directory in os.walk(self.data_path):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                regex = r"-"+self.mode+"_subsampled\.nii\.gz$"
                if re.search(regex, input_file):
                    pat_code = input_file.\
                                rsplit('-'+self.mode+'_subsampled.nii.gz')
                    patient_code = pat_code[0].rsplit('/', 1)[1]
                    if patient_code in self.patients_dict:
                        if patient_code not in self.valid_patients:
                            self.train_patients.append(patient_code)

        train_X = []
        train_Y = []
        valid_X = []
        valid_Y = []
        valid_pat_code = []
        # Add training data to train_X and validation to valid_X
        for directory in os.walk(self.data_path):
            # Walk inside the directory
            for file in directory[2]:
                # Match all files ending with 'regex'
                input_file = os.path.join(directory[0], file)
                regex = r"-"+self.mode+"_subsampled\.nii\.gz$"
                if re.search(regex, input_file):
                    pat_code = input_file.rsplit('-'+self.mode+'_subsampled.nii.gz')
                    patient_code = pat_code[0].rsplit('/', 1)[1]
                    feature_selected_image = []
                    if patient_code in self.train_patients or patient_code in \
                            self.valid_patients:
                        mri_image = nb.load(input_file)
                        mri_image = mri_image.get_data()
                        mri_image = mri_image.flatten()

                        if scaling == True:
                            mri_image = pre.scale(mri_image, copy=False)

                        feature_selected_image = np.take(mri_image, features)
                        #print("Feature selected: ",
                        # len(feature_selected_image))
                        if len(feature_selected_image) == 0:
                            raise ValueError('Zero selected features')

                    if patient_code in self.train_patients:
                        train_X.append(feature_selected_image)
                        train_Y.append(self.patients_dict[patient_code])
                    elif patient_code in self.valid_patients:
                        valid_pat_code.append(patient_code)
                        valid_X.append(feature_selected_image)
                        valid_Y.append(self.patients_dict[patient_code])

        print("Train:", len(train_X))
        print("Valid:", len(valid_X))

        ntrain = np.array(train_X)
        ntrain_labels = np.array(train_Y)

        #np.save('./train/'+self.mode+"_train", ntrain)
        #np.save('./train/'+self.mode+"_train_labels", ntrain_labels)

        nvalid  = np.array(valid_X)
        nvalid_labels = np.array(valid_Y)

        #np.save('./valid/'+self.mode+"_valid", nvalid)
        #np.save('./valid/'+self.mode+"_valid_labels", nvalid_labels)

        # Training grid search parameters
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10]}]

        classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, n_jobs=3,
                                  scoring='accuracy')
        classifier.fit(ntrain,ntrain_labels)

        print("Best parameters set found: ", classifier.best_params_)
        #print("Storing the self.model...")
        #joblib.dump(classifier, self.mode+'_classifier.pkl')

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
        return valid_pat_code, pred