import os
from os import path
import re
import sys
import random
import numpy as np
from sklearn import svm
import nibabel as nb
import pickle
import argparse

import sklearn.preprocessing as pre
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from dementia_prediction.classical.models.svm_class import SVM
from dementia_prediction.config_wrapper import Config

def get_masked_features(brain_mask_path, fisher_score_all_voxels_path,
                        percent):

    # Get the brain mask and find voxel indices inside brain
    mri_image_mask = nb.load(brain_mask_path)
    mri_image_mask = mri_image_mask.get_data()
    mri_image_mask = mri_image_mask.flatten()
    non_zero_indices = np.nonzero(mri_image_mask)[0] 

    # Load fisher scores of all voxels and store only brain voxels scores
    filep = open(fisher_score_all_voxels_path, 'rb')
    fisher_scores_all = pickle.load(filep)
    non_zero_scores = np.take(fisher_scores_all, non_zero_indices)
    print("Only", len(non_zero_scores), "non zero voxels")

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
    parser = argparse.ArgumentParser(description="Run SVM")
    parser.add_argument("paramfile", type=str,
                        help='Path to the parameter file')
    args = parser.parse_args()
    config.parse(path.abspath(args.paramfile))
    params = config.config.get('parameters')
    brain_mask_path = params['brain_mask']
    fisher_scores_all = params['fisher_score_all']
    print("Input fisher scores: ", fisher_scores_all)
    for percent in [0.1, 0.3, 0.5, 0.8, 1]:
        print("Percentage of features chosen: "+str(percent*100))
        features, scores = get_masked_features(brain_mask_path,
                                               fisher_scores_all,
                                               percent)
        print("Length of selected features: ", len(features))
        mask_features_out =  params['out']+'Top_'+\
                             str(percent*100)+'_features.pkl'
        mask_fisher_out = params['out']+'Top_'+\
                             str(percent*100)+'_scores.pkl'
        with open(mask_features_out,'wb') as p_filep:
            pickle.dump(features, p_filep)
        with open(mask_fisher_out, 'wb') as p_filep:
            pickle.dump(scores, p_filep)
        svm = SVM(params)
        svm.train_and_predict(False, features)


if __name__ == "__main__":
    main()
