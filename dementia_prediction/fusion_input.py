"""
This module contains functions to handle the input data to the CNN model.
"""

import math
import random
import numpy as np
import nibabel as nb


class FusionDataInput:
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
        self.mode_folders = [params['mode_folder'+str(i)] for i in range(1, 4)]
        self.modalities = {0: params['cnn']['mode1'],
                           1: params['cnn']['mode2'],
                           2: params['cnn']['mode3']
                           }

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

    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = [np.array([], np.float) for i in range(0,
                                                              len(self.modalities))]
        batch_labels = np.zeros((self.params['batch_size'], self.num_classes))

        iterate = 0
        batch_files = []
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
                for index in range(0, len(self.modalities)):
                    mode_file = self.mode_folders[index]+\
                                filename.rsplit('/',1)[1]
                    mri_image = nb.load(mode_file)
                    #print(self.name+" "+mode_file+" "+str(class_label), flush=True)
                    #mri_image = mri_image.get_data().flatten()
                    #mri_image = self.normalize(mri_image)
                    mri_image = mri_image.get_data()
                    mri_image = np.reshape(mri_image, [1, self.params['depth'],
                                                       self.params['height'],
                                                       self.params['width'], 1])

                    if len(batch_images[index]) == 0:
                        batch_images[index] = mri_image
                    else:
                        batch_images[index] = np.append(batch_images[index],
                                                       mri_image,
                                                  axis=0)
                batch_labels[iterate][class_label] = 1 #class_label: 0 - NC, 1 - MCI, 2 - AD
                iterate += 1
            batch_files += class_files
        return batch_files, batch_images, batch_labels
