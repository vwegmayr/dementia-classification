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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class SVM:

    def __init__(self, params):
        self.data_path = params['data_path']
        self.patients_dict = pickle.load(open(params['patient_dict'], 'rb'))
        self.valid_patients = pickle.load(open(params['valid_dict'], 'rb'))
        self.train_patients = pickle.load(open(params['train_dict'], 'rb'))
        self.params = params
        np.random.seed(1)

    def train_and_predict(self, scaling, features):
        """
        Args:
            scaling: Boolean argument to normalize the input image 
            features: features to be selected from the input image.

        Returns: SVM classifier
        """

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
                regex = r"" + self.params['regex'] + "$"
                #print(file)
                if re.search(regex, input_file):
                    pat_code = input_file.rsplit(self.params['split_on'])
                    patient_code = pat_code[0].rsplit('/', 1)[1]
                    #print(patient_code)
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
        for model in ['svm', 'rf']:
            classifier = 0
            if model == 'svm':
                classifier = svm.SVC(C=1, kernel='linear', class_weight='balanced')
            elif model == 'rf':
                classifier = RandomForestClassifier(n_estimators=1000, n_jobs=5)
            classifier.fit(ntrain,ntrain_labels)

            # Predict on validation data
            pred = classifier.predict(nvalid)
            correct = 0
            for i in range(0, len(pred)):
                if pred[i] == nvalid_labels[i]:
                    correct += 1

            print("Accuracy: ",(correct/len(pred)))
            print("Confusion matrix:", confusion_matrix(nvalid_labels, pred))
        
