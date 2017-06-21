"""
This module contains functions to handle the input data to the CNN model.
"""

import math
import random
import numpy as np
import nibabel as nb


class DataInput:
    """
    This class provides helper functions to manage the input datasets.
    Initialize this class with the required parameter file and the dataset
    as a tuple of filenames. If the data is not normalized, you can pass the
    mean and variance of the dataset to normalize.
    """
    def __init__(self, params, data, name, mean=0, var=0):
        self.data = params['cnn']
        self.s_files = data[0]
        self.p_files = data[1]
        self.s_batch_index = 0
        self.p_batch_index = 0
        self.name = name
        self.mean = mean
        self.var = var

    def normalize(self, mri_image):
        norm_image = []
        if self.mean == 0:
            mean = mri_image.mean()
            stddev = mri_image.std()
            #adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            norm_image = (mri_image - mean) / stddev
        else:
            for x,y,z in zip(mri_image, self.mean, self.var):
                if z == 0:
                    norm_image.append(0)
                else:
                    norm_image.append((x-y)/z)
            #print(norm_image)
        return  np.reshape(norm_image, [1, self.data['depth'],
                                           self.data['height'],
                                           self.data['width'], 1])
    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = np.array([], np.float)
        batch_labels = np.zeros((self.data['batch_size'], 2))

        iterate = 0
        start = self.s_batch_index
        end = start + int(self.data['batch_size']/2)
        self.s_batch_index += int(self.data['batch_size']/2)

        if end > len(self.s_files):
            # Reached end of epoch
            shuffle_indices = list(range(len(self.s_files)))
            random.shuffle(shuffle_indices)
            self.s_files = [self.s_files[i] for i in shuffle_indices]
            start = 0
            end = int(self.data['batch_size']/2)
            self.s_batch_index = int(self.data['batch_size']/2)
        #print("Batch patients:")
        for i in range(start, end):
            mri_image = nb.load(self.s_files[i])
            #print(self.name+" "+self.s_files[i]+" 0", flush=True)
            #mri_image = mri_image.get_data().flatten()
            #mri_image = self.normalize(mri_image)
            mri_image = mri_image.get_data()
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])

            if len(batch_images) == 0:
                batch_images = mri_image
            else:
                batch_images = np.append(batch_images, mri_image, axis=0)
            batch_labels[iterate][0] = 1 #1 - demented, 0 - stable
            iterate += 1
        
        start = self.p_batch_index
        end = start + int(self.data['batch_size']/2)
        self.p_batch_index += int(self.data['batch_size']/2)

        if end > len(self.p_files):
            # Reached end of epoch
            shuffle_indices = list(range(len(self.p_files)))
            random.shuffle(shuffle_indices)
            self.p_files = [self.p_files[i] for i in shuffle_indices]
            start = 0
            end = int(self.data['batch_size']/2)
            self.p_batch_index = int(self.data['batch_size']/2)
        #print("Batch patients:")
        for i in range(start, end):
            mri_image = nb.load(self.p_files[i])
            #print(self.name+" "+self.p_files[i]+" 1",flush=True)
            #mri_image = mri_image.get_data().flatten()
            #mri_image = self.normalize(mri_image)
            mri_image = mri_image.get_data()
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            if len(batch_images) == 0:
                batch_images = mri_image
            else:
                batch_images = np.append(batch_images, mri_image, axis=0)

            batch_labels[iterate][1] = 1
            iterate += 1

        # print("Batch Index: " + str(self.batch_index))

        return batch_images, batch_labels
