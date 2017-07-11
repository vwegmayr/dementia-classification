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
    as a tuple of filenames for each class.
    """
    def __init__(self, params, data, name, mean=0, var=0):
        self.params = params['cnn']
        self.num_classes = len(data)
        self.files = [[] for i in range(0, self.num_classes)]
        self.files[0] = data[0]
        self.files[1] = data[1]
        self.files[2] = data[2]
        self.batch_index = [0 for i in range(0, self.num_classes)]
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
        return  np.reshape(norm_image, [1, self.params['depth'],
                                           self.params['height'],
                                           self.params['width'], 1])
    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = np.array([], np.float)
        batch_labels = np.zeros((self.params['batch_size'], self.num_classes))

        iterate = 0
        for class_label in range(0, self.num_classes):
            start = self.batch_index[class_label]
            end = start + int(self.params['batch_size']/self.num_classes)
            self.batch_index[class_label] += int(self.params['batch_size']/self.num_classes)
            print("Start:", start, "End:", end)
            if end > len(self.files[class_label]):
                # Reached end of epoch
                shuffle_indices = list(range(len(self.files[class_label])))
                random.shuffle(shuffle_indices)
                self.files[class_label] = [self.files[class_label][i] for i in shuffle_indices]
                start = 0
                end = int(self.params['batch_size']/self.num_classes)
                self.batch_index[class_label] = int(self.params['batch_size']/self.num_classes)
            for i in range(start, end):
                mri_image = nb.load(self.files[class_label][i])
                print(self.name+" "+self.files[class_label][i]+" "+str(class_label), flush=True)
                #mri_image = mri_image.get_data().flatten()
                #mri_image = self.normalize(mri_image)
                mri_image = mri_image.get_data()
                mri_image = np.reshape(mri_image, [1, self.params['depth'],
                                                   self.params['height'],
                                                   self.params['width'], 1])

                if len(batch_images) == 0:
                    batch_images = mri_image
                else:
                    batch_images = np.append(batch_images, mri_image, axis=0)
                batch_labels[iterate][class_label] = 1 #class_label: 0 - NC, 1 - MCI, 2 - AD
                iterate += 1
        
        return batch_images, batch_labels
