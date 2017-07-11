import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)


modeB = 'T1_brain'
modeA = 'T1_brain'
print(modeA, "10%")
scoresA = pickle.load(open('./fisher_10/'+modeA+'_fisher_score_all.pkl', 'rb'))
scoresB = pickle.load(open(
    './fisher_10/'+modeB+'_fisher_score_robust_parallel_all_newbins.pkl',
    'rb'))

# Get the x and y axis ranges masking the inf and nans if any
minA = np.min(np.ma.masked_invalid(scoresA))
minB = np.min(np.ma.masked_invalid(scoresB))
maxA = np.max(np.ma.masked_invalid(scoresA))
maxB = np.max(np.ma.masked_invalid(scoresB))


featA = pickle.load(open(
    './features/mask/naive/'+modeB+'_features_30.0_naive_mask'
                                                        '.pkl',
                         'rb'))
featB = pickle.load(open(
    './features/mask/robust/'+modeA
    +'_features_30.0_robust_mask.pkl',
    'rb'))
print(len(featA), len(featB))

#common_feat = list(set(featA) & set(featB))
color_plot_X = []
color_plot_Y = []
#print("Common feat cnt: ", len(common_feat))

for i in range(0, 897600):
    if i in featA and i in featB:
        if np.isinf(scoresA[i]):
            color_plot_X.append(maxA)
        else:
            color_plot_X.append(scoresA[i])

        if np.isinf(scoresB[i]):
            color_plot_Y.append(maxB)
        else:
            color_plot_Y.append(scoresB[i])

fig, ax = plt.subplots()

brain_mask_path= '/home/rams/4_Sem/Thesis/Data/T1_brain_subsampled/CON018/CON018-T1_brain_mask_subsampled.nii.gz'
mri_image_mask = nb.load(brain_mask_path)
mri_image_mask = mri_image_mask.get_data()
mri_image_mask = mri_image_mask.flatten()
non_zero_indices = np.nonzero(mri_image_mask)[0]

non_zero_scoresA = np.take(scoresA, non_zero_indices)
non_zero_scoresB = np.take(scoresB, non_zero_indices)

ax.scatter(non_zero_scoresA, non_zero_scoresB, s=5)
ax.scatter(color_plot_X, color_plot_Y, c='r', s=5)

# Set the ranges of X and Y axes
ax.set_xlim([minA, maxA])
ax.set_ylim([minB, maxB])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
print("Limits: ", lims)

# now plot both limits against eachother
#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
plt.xticks(np.arange(minA, maxA, 0.01))
plt.yticks(np.arange(minB, maxB, 0.1))
plt.xlabel(modeA+' Naive Fisher Score')
plt.ylabel(modeB+' Robust Fisher Score')
plt.title('T1 Fisher Scores Robust vs Naive')
ax.legend(loc='upper right')
#plt.grid(True)
plt.show()
fig.savefig(modeA+"Rob_Naive.png")