import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
import argparse
import os
from os import path
import scipy.stats as sc
from dementia_prediction.config_wrapper import Config
from sklearn.preprocessing import normalize

config = Config()
parser = argparse.ArgumentParser(description="Generate Scatter Plots")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams.update({'font.size': 12})
scoresA = pickle.load(open(params['naive_scores'],  'rb'))
scoresB = pickle.load(open(params['robust_scores'], 'rb'))

allscoresA = pickle.load(open(params['naive_all_scores'],  'rb'))
allscoresB = pickle.load(open(params['robust_all_scores'], 'rb'))
IMG_SIZE = params['depth'] * params['height'] * params['width']

# Get the x and y axis ranges masking the inf and nans if any
minA = np.min(np.ma.masked_invalid(allscoresA))
minB = np.min(np.ma.masked_invalid(allscoresB))
maxA = np.max(np.ma.masked_invalid(allscoresA))
maxB = np.max(np.ma.masked_invalid(allscoresB))

brain_mask_path= params['mask_path']
mri_image_mask = nb.load(brain_mask_path)
mri_image_mask = mri_image_mask.get_data()
mri_image_mask = mri_image_mask.flatten()
non_zero_indices = np.nonzero(mri_image_mask)[0]

non_zero_scoresA = np.take(allscoresA, non_zero_indices)
non_zero_scoresB = np.take(allscoresB, non_zero_indices)
norm_nonzero_A = non_zero_scoresA/np.linalg.norm(non_zero_scoresA)
norm_nonzero_B = non_zero_scoresB/np.linalg.norm(non_zero_scoresB)
print("Pearson", sc.pearsonr(norm_nonzero_A, norm_nonzero_B))
print("Spearmanr", sc.spearmanr(non_zero_scoresA, non_zero_scoresB))
