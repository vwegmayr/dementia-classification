"""
This module contains functions to handle the input data to the CNN model.
"""

import math
import random
import numpy as np
import nibabel as nb
import re
import pickle

class DataInput:
    """
    This class provides helper functions to manage the input datasets.
    Initialize this class with the required parameter file and the dataset
    as a tuple of filenames.
    """
    def __init__(self, params, data, name):
        self.data = params['cnn']
        self.pos = data[0]
        self.neg = data[1]
        self.pos_batch_index = 0
        self.neg_batch_index = 0
        self.name=name
        self.CBF_features = params['CBF_feat']
        self.T1_features = params['T1_feat']
        self.DTI_MO_features = params['DTI_MO_feat']
        self.DTI_MD_features = params['DTI_MD_feat']
        self.DTI_FA_features = params['DTI_FA_feat']

    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = np.array([], np.float)
        batch_labels = np.zeros((self.data['batch_size'], 2))

        iterate = 0
        start = self.pos_batch_index
        end = start + int(self.data['batch_size']/2)
        self.pos_batch_index += int(self.data['batch_size']/2)

        if end > len(self.pos):
            # Reached end of epoch
            shuffle_indices = list(range(len(self.pos)))
            random.shuffle(shuffle_indices)
            self.pos = [self.pos[i] for i in shuffle_indices]
            start = 0
            end = int(self.data['batch_size']/2)
            self.pos_batch_index = int(self.data['batch_size']/2)
        print("Batch patients:")
        for i in range(start, end):
            mri_image = nb.load(self.pos[i])
            print(self.name+" "+self.pos[i]+" 0", flush=True)
            mri_image = mri_image.get_data()
            mri_image = mri_image.flatten()
            features = []
            if re.search(r"-CBF_subsampled\.nii\.gz$", self.pos[i]):
                features = pickle.load(open(self.CBF_features, 'rb'))
            elif re.search(r"-T1_brain_subsampled\.nii\.gz$", self.pos[i]):
                features = pickle.load(open(self.T1_features, 'rb'))
            elif re.search(r"-DTI_MO_subsampled\.nii\.gz$", self.pos[i]):
                features = pickle.load(open(self.DTI_MO_features, 'rb'))
            elif re.search(r"-DTI_MD_subsampled\.nii\.gz$", self.pos[i]):
                features = pickle.load(open(self.DTI_MD_features, 'rb'))
            elif re.search(r"-DTI_FA_subsampled\.nii\.gz$", self.pos[i]):
                features = pickle.load(open(self.DTI_FA_features, 'rb'))
            # print("Shape of mri: " + str(mri_image.shape) + " type: "
            # "" + str(mri_image.dtype) + " Mean: " + str(mri_image.mean()))
            feature_selected_image = np.take(mri_image, features)
            #print(len(feature_selected_image))
            if len(batch_images) == 0:
                batch_images = feature_selected_image
            else:
                batch_images = np.vstack((batch_images,
                                          feature_selected_image))
            #print(batch_images.shape)
            batch_labels[iterate][0] = 1 #1 - demented, 0 - stable
            iterate += 1
        
        start = self.neg_batch_index
        end = start + int(self.data['batch_size']/2)
        self.neg_batch_index += int(self.data['batch_size']/2)

        if end > len(self.neg):
            # Reached end of epoch
            shuffle_indices = list(range(len(self.neg)))
            random.shuffle(shuffle_indices)
            self.neg = [self.neg[i] for i in shuffle_indices]
            start = 0
            end = int(self.data['batch_size']/2)
            self.neg_batch_index = int(self.data['batch_size']/2)
        #print("Batch patients:")
        for i in range(start, end):
            mri_image = nb.load(self.neg[i])
            print(self.name+" "+self.neg[i]+" 1",flush=True)
            mri_image = mri_image.get_data()
            mri_image = mri_image.flatten()
            features = []
            if re.search(r"-CBF_subsampled\.nii\.gz$", self.neg[i]):
                features = pickle.load(open(self.CBF_features, 'rb'))
            elif re.search(r"-T1_brain_subsampled\.nii\.gz$", self.neg[i]):
                features = pickle.load(open(self.T1_features, 'rb'))
            elif re.search(r"-DTI_MO_subsampled\.nii\.gz$", self.neg[i]):
                features = pickle.load(open(self.DTI_MO_features, 'rb'))
            elif re.search(r"-DTI_MD_subsampled\.nii\.gz$", self.neg[i]):
                features = pickle.load(open(self.DTI_MD_features, 'rb'))
            elif re.search(r"-DTI_FA_subsampled\.nii\.gz$", self.neg[i]):
                features = pickle.load(open(self.DTI_FA_features, 'rb'))
            
            feature_selected_image = np.take(mri_image, features)

            # print("Shape of mri: " + str(mri_image.shape) + " type: "
            # "" + str(mri_image.dtype) + " Mean: " + str(mri_image.mean()))

            if len(batch_images) == 0:
                batch_images = feature_selected_image
            else:
                batch_images = np.vstack((batch_images, 
                                          feature_selected_image))
            #print(batch_images.shape)
            batch_labels[iterate][1] = 1
            iterate += 1

        # print("Batch Index: " + str(self.batch_index))

        return batch_images, batch_labels
