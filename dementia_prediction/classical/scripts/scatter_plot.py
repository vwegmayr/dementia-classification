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

from dementia_prediction.config_wrapper import Config

config = Config()
parser = argparse.ArgumentParser(description="Generate Scatter Plots")
parser.add_argument("paramfile", type=str, help='Path to the parameter file')
args = parser.parse_args()
config.parse(path.abspath(args.paramfile))
params = config.config.get('parameters')

font = {'family' : 'normal',
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


featA = pickle.load(open(params['naive_features'], 'rb'))
featB = pickle.load(open(params['robust_features'], 'rb'))
print("Number of features: Naive:", len(featA), "Robust:", len(featB))

common_feat = list(set(featA) & set(featB))
'''
print(len(common_feat))
with open('./common_features_rob_naive.pkl', 'wb') as filep:
    pickle.dump(common_feat, filep)
'''
color_plot_X = []
color_plot_Y = []
for i in range(0, len(featA)):
    try:
        # Check if feature A is in feature B, If not raise error
        featureB_index = featB.tolist().index(featA[i])
        # If present, then store it for plotting
        if np.isinf(scoresA[i]):
            color_plot_X.append(maxA)
            print("Score is infinite")
        else:
            color_plot_X.append(scoresA[i])

        if np.isinf(scoresB[featureB_index]):
            color_plot_Y.append(maxB)
            print("Score is infinite")
        else:
            color_plot_Y.append(scoresB[featureB_index])
    except ValueError:
        continue
print("Common Features:", len(color_plot_X))
fig, ax = plt.subplots()
brain_mask_path= params['mask_path']
mri_image_mask = nb.load(brain_mask_path)
mri_image_mask = mri_image_mask.get_data()
mri_image_mask = mri_image_mask.flatten()
non_zero_indices = np.nonzero(mri_image_mask)[0]

non_zero_scoresA = np.take(allscoresA, non_zero_indices)
non_zero_scoresB = np.take(allscoresB, non_zero_indices)

ax.scatter(non_zero_scoresA, non_zero_scoresB, s=5)
ax.scatter(color_plot_X, color_plot_Y, c='r', s=5)
#print("Below two rows should be same. Sanity check:")
#print(color_plot_X[0], color_plot_Y[0])
#print(non_zero_scoresA[0], non_zero_scoresB[0])

# Set the ranges of X and Y axes
minB = 0
minA = 0
'''
# UHG T1, T2, DTI FA
maxA = 0.06 
maxB = 0.75 
'''
# OASIS T1
maxB = 2
ax.set_xlim([minA, maxA])
ax.set_ylim([minB, maxB])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
print("Limits: ", lims)

# now plot both limits against eachother
#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
'''
# UHG T1, T2, DTI FA
plt.xticks(np.arange(minA, maxA+0.01, 0.01))
plt.yticks(np.arange(minB, maxB+0.1, 0.1))
'''
# OASIS
plt.xticks(np.arange(minA, maxA+0.05, 0.05))
plt.yticks(np.arange(minB, maxB+0.25, 0.25))
plt.xlabel(params['xlabel'])
plt.ylabel(params['ylabel'])
plt.title(params['title'], fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
#plt.grid(True)
#plt.show()
fig.savefig(params['fig_name'])
