"""
This module contains functions to handle the input data to the CNN model.
"""

import math
import random
import numpy as np
import nibabel as nb
import scipy.ndimage.interpolation as sni

class DataInput:
    """
    This class provides helper functions to manage the input datasets.
    Initialize this class with the required parameter file and the dataset
    as a tuple of filenames for each class.
    """
    def __init__(self, params, data, name, mean=0, var=0):
        self.params = params['cnn']
        self.num_classes = len(data)
        self.files = [data[i] for i in range(0, self.num_classes)]
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
    def shuffle(self):
        for class_label in range(0, self.num_classes):
            shuffle_indices = list(range(len(self.files[class_label])))
            random.shuffle(shuffle_indices)
            self.files[class_label] = [self.files[class_label][i] for i in shuffle_indices]

    def rotate(self, filename, direction):
        angle_rot = random.uniform(-5, 5)
        mri_image = nb.load(filename).get_data()
        if direction == 'x':
            return sni.rotate(mri_image, angle_rot, (0,1), reshape=False)
        if direction == 'y':
            return sni.rotate(mri_image, angle_rot, (0,2), reshape=False)
        if direction == 'z':
            return sni.rotate(mri_image, angle_rot, (1,2), reshape=False)

    def translate(self, filename, direction):
        pixels = random.uniform(-4, 4)
        mri_image = nb.load(filename).get_data()
        if direction == 'x':
            return sni.shift(mri_image, [pixels, 0, 0], mode='nearest')
        if direction == 'y':
            return sni.shift(mri_image, [0, pixels, 0], mode='nearest')
        if direction == 'z':
            return sni.shift(mri_image, [0, 0, pixels], mode='nearest')

    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = np.array([], np.float)
        batch_labels = np.zeros((self.params['batch_size'], self.num_classes))

        iterate = 0
        batch_files = []
        # Increment batch_size/num_class for each class
        inc = int(self.params['batch_size']/self.num_classes)
        inc = 1
        # If unbalanced, then increment the images of a specific class, say 0
        unbalanced = self.params['batch_size']%self.num_classes
        batch_order = []
        for i in range(0, self.num_classes):
            batch_order += [i for j in range(0,int(self.params['batch_size']/self.num_classes))]
        random.shuffle(batch_order)
        for class_label in batch_order:
            start = self.batch_index[class_label]
            class_files = []
            if unbalanced != 0 and class_label == 0:
                end = start + inc + 1 
                self.batch_index[class_label] += inc + 1 
            else:
                end = start + inc
                self.batch_index[class_label] += inc 

            if end > len(self.files[class_label]):
                # Reached end of epoch
                class_files = [self.files[class_label][i] for i in range(start, len(self.files[class_label]))]
                left_files = end - len(self.files[class_label])
                shuffle_indices = list(range(len(self.files[class_label])))
                random.shuffle(shuffle_indices)
                self.files[class_label] = [self.files[class_label][i] for i in shuffle_indices]
                start = 0
                end = left_files
                self.batch_index[class_label] = left_files
            for i in range(start, end):
                class_files.append(self.files[class_label][i])
            for filename in class_files:
                # For augmentation
                mri_image = []
                if 'rot' in filename:
                    split_filename = filename.split('rot')
                    mri_image = self.rotate(split_filename[0],
                                            split_filename[1])
                elif 'trans' in filename:
                    split_filename = filename.split('trans')
                    mri_image = self.translate(split_filename[0],
                                            split_filename[1])

                else:
                    mri_image = nb.load(filename)
                    #mri_image = mri_image.get_data().flatten()
                    #mri_image = self.normalize(mri_image)
                    mri_image = mri_image.get_data()
                #print(self.name+" "+filename+" "+str(class_label), flush=True)
                mri_image = np.reshape(mri_image, [1, self.params['depth'],
                                                   self.params['height'],
                                                   self.params['width'], 1])

                if len(batch_images) == 0:
                    batch_images = mri_image
                else:
                    batch_images = np.append(batch_images, mri_image, axis=0)
                batch_labels[iterate][class_label] = 1 #class_label: 0 - NC, 1 - MCI, 2 - AD
                iterate += 1
            batch_files += class_files
        
        return batch_files, batch_images, batch_labels
