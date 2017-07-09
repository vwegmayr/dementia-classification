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
    as a tuple of filenames.
    """
    def __init__(self, params, data, name):
        self.data = params['cnn']
        self.pos = data[0]
        self.neg = data[1]
        self.pos_batch_index = 0
        self.neg_batch_index = 0
        self.name=name
        self.T1_folder=params['T1_folder']
        self.DTI_folder=params['DTI_folder']

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
            # Taking CBF image as input
            mri_image = nb.load(self.pos[i])
            print(self.name+" "+self.pos[i]+" 0", flush=True)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            #adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            # Appending T1 image
            temp_image = mri_image
            patient = self.pos[i].rsplit('/', 1)[1]
            pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
            T1_filename = self.T1_folder + pat_code + '/' + pat_code
            T1_filename += '-T1_brain_subsampled.nii.gz'
            print(self.name+" "+T1_filename+" "+" 0", flush=True)
            mri_image = nb.load(T1_filename)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            temp_image = np.append(temp_image, mri_image, axis = 4)

            #Appending DTI image
            patient = self.pos[i].rsplit('/', 1)[1]
            pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
            DTI_filename = self.DTI_folder + pat_code 
            DTI_filename += '-DTI_MO_subsampled.nii.gz'
            print(self.name+" "+DTI_filename+" "+" 0", flush=True)
            mri_image = nb.load(DTI_filename)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            #adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            temp_image = np.append(temp_image, mri_image, axis = 4)
            #print("Shape:", batch_images.shape)
            if len(batch_images) == 0:
                batch_images = temp_image
            else:
                batch_images = np.append(batch_images, temp_image, axis=0)
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
            # Taking CBF image as input
            mri_image = nb.load(self.neg[i])
            print(self.name+" "+self.neg[i]+" 1",flush=True)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            #adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])

            # print("Shape of mri: " + str(mri_image.shape) + " type: "
            # "" + str(mri_image.dtype) + " Mean: " + str(mri_image.mean()))

            # Appending T1 image
            temp_image = mri_image
            patient = self.neg[i].rsplit('/', 1)[1]
            pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
            T1_filename = self.T1_folder + pat_code + '/' + pat_code
            T1_filename += '-T1_brain_subsampled.nii.gz'
            print(self.name+" "+T1_filename+" "+" 1", flush=True)
            mri_image = nb.load(T1_filename)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            temp_image = np.append(temp_image, mri_image, axis = 4)

            #Appending DTI image
            patient = self.neg[i].rsplit('/', 1)[1]
            pat_code = patient.split('-CBF_subsampled.nii.gz')[0]
            DTI_filename = self.DTI_folder + pat_code 
            DTI_filename += '-DTI_MO_subsampled.nii.gz'
            print(self.name+" "+DTI_filename+" "+" 1", flush=True)
            mri_image = nb.load(DTI_filename)
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            #adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])
            temp_image = np.append(temp_image, mri_image, axis = 4)
            #print("Shape:", batch_images.shape)
            if len(batch_images) == 0:
                batch_images = temp_image
            else:
                batch_images = np.append(batch_images, temp_image, axis=0)
            batch_labels[iterate][1] = 1
            iterate += 1

        # print("Batch Index: " + str(self.batch_index))

        return batch_images, batch_labels