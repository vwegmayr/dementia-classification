import os
from os import path
import re
import sys
import random
import numpy as np
from sklearn import svm
import nibabel as nb
import pickle
from collections import defaultdict

import sklearn.preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from svm_class import SVM
from dementia_prediction.config_wrapper import Config

mode = 'CBF'

def get_masked_features(brain_mask_path, fisher_score_all_voxels_path,
                        percent):

    # Get the brain mask and find voxel indices inside brain
    mri_image_mask = nb.load(brain_mask_path)
    mri_image_mask = mri_image_mask.get_data()
    mri_image_mask = mri_image_mask.flatten()
    non_zero_indices = np.nonzero(mri_image_mask)[0] # 182200 indices

    # Load fisher scores of all voxels and store only brain voxels scores
    filep = open(fisher_score_all_voxels_path, 'rb')
    fisher_scores_all = pickle.load(filep)
    non_zero_scores = np.take(fisher_scores_all, non_zero_indices) # 188200

    # Sort the brain voxels scores and get the sorted indices of brain voxels
    sorted_fisher_indices = sorted(range(len(non_zero_scores)),
                               key=lambda k: non_zero_scores[k], reverse=True)
    significant_len = int(percent*len(non_zero_scores))
    top_indices_internal = sorted_fisher_indices[:significant_len]

    # percent*188200 indices and score
    final_indices =  np.take(non_zero_indices, top_indices_internal)
    final_scores = np.take(fisher_scores_all, final_indices)

    return final_indices, final_scores


def main():

    config = Config()
    param_file = sys.argv[1]
    config.parse(path.abspath(param_file))
    paths = config.config.get('paths')

    brain_mask_path = paths['brain_mask']
    patients_dict = pickle.load(open(path.abspath(paths['patient_path']),
                                     'rb'))
    valid_correct = pickle.load(open(path.abspath(paths['valid_path']),
                                     'rb'))
    print(valid_correct)
    valid_patients = defaultdict(int)
    for mode in ['CBF', 'T1_brain', 'DTI_FA']:
        print("Mode: ", mode)
        fisher_scores_all = paths['fisher_all']+mode+'_fisher_score_robust_parallel_all_newbins.pkl'
        print("Input fisher scores: ", fisher_scores_all)
        for percent in [0.3]:

            print("Percentage of features chosen: "+str(percent*100))
            mask_features_out = paths['out_features_path'] + 'mask/robust/' + \
                                mode + '_features_' + str(percent * 100) + \
                                '_robust_mask.pkl'
            mask_fisher_out = paths[
                                  'out_fisher_path'] + mode + '_fisher_' + str(
                percent * 100) + \
                              '_robust_mask.pkl'
            if not os.path.exists(mask_fisher_out):
                features, scores = get_masked_features(brain_mask_path,
                                                       fisher_scores_all,
                                                       percent)
                print("Length of selected features: ", len(features))

                with open(mask_features_out,'wb') as p_filep:
                    pickle.dump(features, p_filep)
                with open(mask_fisher_out, 'wb') as p_filep:
                    pickle.dump(scores, p_filep)
            else:
                with open(mask_features_out,'rb') as p_filep:
                    features = pickle.load(p_filep)
            svm = SVM(paths['data'], mode, path.abspath(paths['valid_path']),
                      path.abspath(paths['patient_path']))
            valid_pat_code, prediction = svm.train_and_predict(False, features)
            # If prediction is 1 add 1, otherwise add -1
            for patient, pred in zip(valid_pat_code, prediction):
                if pred == 1:
                    valid_patients[patient] += 1
                else:
                    valid_patients[patient] -= 1
                print("patient: ", patient, valid_patients[patient])
    correct = 0
    for code in valid_correct:
        if patients_dict[code] == 1 and valid_patients[code] > 0:
            correct += 1
        elif patients_dict[code] == 0 and valid_patients[code] < 0:
            correct += 1
        print("Code:", code, "Actual:", patients_dict[code], "Pred:",
              valid_patients[code])
    print("Accuracy: ", (correct / len(valid_patients)))
if __name__ == "__main__":
    main()
