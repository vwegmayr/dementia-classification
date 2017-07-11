import pickle
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

mode = 'DTI_MO'
#bins = np.linspace(0, 0.07, 10)
#bins = [0, 0.2, 0.4, 0.6, 0.7]
filep = open('./fisher_10/'+mode+'_fisher_score_10_with_test.pkl', 'rb')
#filep = open('./image.pkl', 'rb')
indices = pickle.load(filep)
print(indices[:10])
# the histogram of the data
#no_bins = (max(indices) - min(indices))/(0.01)
#print(no_bins)
#plt.hist(indices, 50, edgecolor='black', alpha=0.75)


# Final working
binwidth = 0.01
n, bins, rects = plt.hist(indices, bins=np.arange(0,
                                                   max(indices) +
                                         binwidth, binwidth),
                          edgecolor='black', alpha=0.75)
for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height+10,
                '%d' % int(height),
                ha='center', va='bottom')
plt.xticks(np.arange(0, max(indices) + binwidth, binwidth))




# add a 'best fit' line
#l = plt.plot(bins, 'r--', linewidth=1)
#binwidth = 0.01
#plt.hist(indices, bins, label='DTI_MO', ec='none')
"""
mode = 'T1_brain'
filep = open('./fisher_10/'+mode+'_fisher_score_10.pkl', 'rb')
indices = pickle.load(filep)
print(len(indices))
print(indices[:10])
plt.hist(indices, bins, label='T1_brain')


mode = 'CBF'
filep = open('./fisher_10/'+mode+'_fisher_score_10.pkl', 'rb')
indices = pickle.load(filep)
print(len(indices))
print(indices[:10])
plt.hist(indices, bins, label='CBF')


'''
filep = open('./'+mode+'_features_10_new.pkl', 'rb')
indices_new = pickle.load(filep)
print(len(set(indices)))
print(len(set(indices) & set(indices_new)))
'''
"""

#plt.setp(patches[0], 'facecolor', 'g')
plt.xlabel('Fisher Score')
plt.ylabel('Number of Voxels')
plt.title('DTI MO All Data Fisher Score Distribution')
plt.legend(loc='upper right')
#plt.grid(True)
plt.show()
