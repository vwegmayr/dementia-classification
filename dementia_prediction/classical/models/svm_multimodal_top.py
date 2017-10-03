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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from dementia_prediction.config_wrapper import Config
if __name__ == '__main__':
        config = Config()
        parser = argparse.ArgumentParser(description="Run SVM multimodal")
        parser.add_argument("paramfile", type=str,
                            help='Path to the parameter file')
        args = parser.parse_args()
        config.parse(path.abspath(args.paramfile))
        params = config.config.get('parameters')

        patients_dict = pickle.load(open(params['patient_dict'], 'rb'))
        valid_patients = pickle.load(open(params['valid_dict'], 'rb'))
        train_patients = pickle.load(open(params['train_dict'], 'rb'))

        IMG_SIZE = params['depth']*params['height']*params['width']

        for percentage in [0.1, 0.3, 0.5, 0.8, 1]:
            print(percentage*100)
            train_X = []
            train_Y = []
            valid_X = []
            valid_Y = []

            features = []
            scores = []
            for i in range(0, 3):
                features_path = params['mode'+str(i)+'_folder'] + 'robust_fisher/Top_' + \
                                str(percentage * 100) + '_features.pkl'
                scores_path = params['mode'+str(i)+'_folder'] +'robust_fisher/Top_'+\
                                         str(percentage*100)+'_scores.pkl'
                features.append(pickle.load(open(features_path, 'rb')))
                scores.append(pickle.load(open(scores_path, 'rb')))
            max_features = len(pickle.load(open(params['mode0_folder']
                                           +'robust_fisher/Top_100_scores.pkl', 'rb')))
            mode_lim = [0 for i in range(0, 3)]
            num_features = 0
            while num_features < percentage*max_features:
                for i in range(0, 3):
                    #print(mode_lim[i], len(scores[i]), num_features)
                    if mode_lim[i] < len(scores[i]):
                        # Picking mode i features
                        second = (i+1)%3
                        third = (i+2)%3
                        if scores[second][mode_lim[second]] <= scores[i][mode_lim[i]] and\
                           scores[third][mode_lim[third]] <= scores[i][mode_lim[i]]:
                            mode_lim[i] += 1
                            num_features += 1
                            break

            print("Total number of features selected:", num_features, "Limits:", flush=True)
            for i in range(0, 3):
                print(mode_lim[i], "out of ", len(features[i]))

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
                                if len(feature_selected_image) == 0:
                                    feature_selected_image = np.take(mri_image,
                                                                     features[i][:mode_lim[i]])
                                else:
                                    feature_selected_image = np.append(feature_selected_image,
                                                       np.take(mri_image, features[i][:mode_lim[i]]))
                            #print("Features selected: ", len(feature_selected_image))
                            if patient_code in train_patients and len(feature_selected_image) > 0:
                                train_X.append(feature_selected_image)
                                train_Y.append(patients_dict[patient_code])
                            if patient_code in valid_patients and len(feature_selected_image) > 0:
                                valid_X.append(feature_selected_image)
                                valid_Y.append(patients_dict[patient_code])


            ntrain = np.array(train_X)
            ntrain_labels = np.array(train_Y)
            print("Train", ntrain.shape)

            nvalid  = np.array(valid_X)
            nvalid_labels = np.array(valid_Y)
            print("Valid", nvalid.shape)
            for model in ['svm', 'rf']:
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
                pred = classifier.predict(nvalid)
                correct = 0
                for i in range(0, len(pred)):
                    if pred[i] == nvalid_labels[i]:
                        correct += 1

                print("Accuracy: ",(correct/len(pred)))
                print("Confusion matrix:", confusion_matrix(nvalid_labels, pred), flush=True)
