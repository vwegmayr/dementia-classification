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
import argparse
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from dementia_prediction.config_wrapper import Config
config = Config()
parser = argparse.ArgumentParser(description="Run SVM multitask")
parser.add_argument("paramfile", type=str,
                    help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')

patients_dict = pickle.load(open(params['patient_dict'], 'rb'))
valid_patients = pickle.load(open(params['valid_dict'], 'rb'))
train_patients = pickle.load(open(params['train_dict'], 'rb'))

for percentage in [0.1, 0.3, 0.5, 0.8, 1]:
    print(percentage, flush=True)
    train_X = [[] for i in range(0, 3)]
    train_Y = [[] for i in range(0, 3)]
    valid_X = [[] for i in range(0, 3)]
    valid_Y = [[] for i in range(0, 3)]
    valid_pat_code = [[] for i in range(0, 3)]

    features = []
    scores = []
    for i in range(0, 3):
        features_path = params['mode'+str(i)+'_folder'] + \
                        'robust_fisher/Top_' + \
                        str(percentage * 100) + '_features.pkl'
        scores_path = params['mode'+str(i)+'_folder'] +'robust_fisher/Top_'+\
                                 str(percentage*100)+'_scores.pkl'
        features.append(pickle.load(open(features_path, 'rb')))
        scores.append(pickle.load(open(scores_path, 'rb')))

    for directory in os.walk(params['data_path']):
        # Walk inside the directory
        for file in directory[2]:
            # Match all files ending with 'regex'
            input_file = os.path.join(directory[0], file)
            regex = r""+params['regex']+"$"
            if re.search(regex, input_file):
                file_name = input_file.rsplit('/', 1)[1]
                patient_code = file_name.split(params['split_on'])[0]
                feature_selected_image = []
                if patient_code in patients_dict:
                    for i in range(0, 3):
                        file_path = params['mode'+str(i)+'_folder']+file_name
                        mri_image = nb.load(file_path)
                        mri_image = mri_image.get_data()
                        mri_image = mri_image.flatten()
                        feature_selected_image = np.take(mri_image,
                                                         features[i])
                        #print("Features selected: ", len(feature_selected_image))
                        if patient_code in train_patients and len(feature_selected_image) > 0:
                            train_X[i].append(feature_selected_image)
                            train_Y[i].append(patients_dict[patient_code])
                        if patient_code in valid_patients and len(feature_selected_image) > 0:
                            valid_X[i].append(feature_selected_image)
                            valid_Y[i].append(patients_dict[patient_code])
                            valid_pat_code[i].append(patient_code)



    for model in ['svm', 'rf']:
        print("Mode:", model, flush=True)
        valid_predictions = defaultdict(int)
        for i in range(0, 3):
            print("Training modality: ", i, flush=True)
            ntrain = np.array(train_X[i])
            ntrain_labels = np.array(train_Y[i])

            print("Train", ntrain.shape)

            classifier = 0
            if model == 'svm':
                tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10]}]
                classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, n_jobs=3,
                                          scoring='accuracy')
            elif model == 'rf':
                tuned_parameters = [{'n_estimators': [10, 100, 1000]}]
                classifier = GridSearchCV(RandomForestClassifier(n_jobs=5), tuned_parameters,
                                        cv=10, n_jobs=2, scoring='accuracy')


            classifier.fit(ntrain,ntrain_labels)

            print("Best parameters set found: ", classifier.best_params_)

            print("Grid scores on development set:")
            means = classifier.cv_results_['mean_test_score']
            stds = classifier.cv_results_['std_test_score']
            print("Mean: ", means, "Std:", stds)
            # Predict on validation data

            nvalid = np.array(valid_X[i])
            print("Valid: ", i, nvalid.shape, flush=True)
            pred = classifier.predict(nvalid)
            correct = 0
            for j in range(0, len(pred)):
                if pred[j] == valid_Y[i][j]:
                    correct += 1

            print("Mode Accuracy: ",(correct/len(pred)))
            print("Mode Confusion matrix:", confusion_matrix(valid_Y[i], pred))
            # If prediction is 1 add 1, otherwise add -1
            for patient, pred in zip(valid_pat_code[i], pred):
                if pred == 1:
                    valid_predictions[patient] += 1
                else:
                    valid_predictions[patient] -= 1
                print("patient: ", patient, valid_predictions[patient])
        correct = 0
        for code in valid_patients:
            if patients_dict[code] == 1 and valid_predictions[code] > 0:
                correct += 1
            elif patients_dict[code] == 0 and valid_predictions[code] < 0:
                correct += 1
            print("Code:", code, "Actual:", patients_dict[code], "Pred:",
                  valid_predictions[code])
        print("Accuracy: ", (correct / len(valid_patients)), flush=True)
