"""
This module contains functions to handle the input data to the CNN model.
"""

import random
import numpy as np
import nibabel as nb


class DataInput:
    """
    This class provides helper functions to manage the input datasets.
    Initialize this class with the required parameter file and the dataset
    as a tuple of filenames and labels.
    """
    def __init__(self, params, data):
        self.data = params['cnn']
        self.filenames = data[0]
        self.labels = data[1]
        self.batch_index = 0

    def next_batch(self):
        """
        This functions retrieves the next batch of the data.

        Returns: (batch_images, batch_labels)

        """
        batch_images = np.array([], np.float)
        batch_labels = np.zeros((self.data['batch_size'], 2))

        iterate = 0
        start = self.batch_index
        end = start + self.data['batch_size']
        self.batch_index += self.data['batch_size']

        if end > len(self.filenames):
            # Reached end of epoch
            shuffle_indices = list(range(len(self.filenames)))
            random.shuffle(shuffle_indices)
            self.filenames = [self.filenames[i] for i in shuffle_indices]
            self.labels = [self.labels[i] for i in shuffle_indices]
            start = 0
            end = self.data['batch_size']
            self.batch_index = self.data['batch_size']

        for i in range(start, end):
            mri_image = nb.load(self.filenames[i])
            mri_image = mri_image.get_data()
            mean = mri_image.mean()
            stddev = mri_image.std()
            # adjusted_stddev = max(stddev, 1.0 / math.sqrt(mri_image.size))
            mri_image = (mri_image - mean) / stddev
            mri_image = np.reshape(mri_image, [1, self.data['depth'],
                                               self.data['height'],
                                               self.data['width'], 1])

            if len(batch_images) == 0:
                batch_images = mri_image
            else:
                batch_images = np.append(batch_images, mri_image, axis=0)

            batch_labels[iterate][self.labels[i]] = 1
            iterate += 1

        # print("Batch Index: " + str(self.batch_index)+" batch labels: "+
        # str(batch_labels))

        return batch_images, batch_labels
