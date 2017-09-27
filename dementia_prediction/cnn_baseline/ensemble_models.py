"""
This module contains the class 'CNN' which enables to build a 3D convolutional
neural network. This neural network convolves along X, Y and Z axis of
the input images to find spatial correlations along all three dimensions.
"""

from datetime import datetime
import time
import math
import tensorflow as tf
import numpy as np
import sys
import pickle
import csv
from dementia_prediction.cnn_utils import CNNUtils

class CNNEnsembleModels:
    """
    This class provides functions to train a 3D Convolutional Neural Network
    model. To train the network and evaluate it, initialize the class with the
    required parameter file and call the function train() with training and
    validation datasets.
    """

    def __init__(self, params):
        self.param = params['cnn']
        self.mode = params['cnn']['mode']
        self.meta_paths = params['cnn']['meta']
        self.cnnutils = CNNUtils(params)

    def evaluation(self, dataset):
        """
        This function evaluates the accuracy of the model
        Args:
            sess: tensorflow session
            eval_op: evaluation operation which calculates number of correct
                    predictions
            dataset: input dataset either train or validation
            images: the images placeholder
            labels: the labels placeholder

        Returns: the accuracy of the 3D CNN model

        """
        correct_predictions = 0
        prediction_data = {}
        total_seen = 0
        pred_out = {}
        class_max_size = 0
        data_size = 0
        for i in range(0, len(dataset.files)):
            data_size += len(dataset.files[i])
            if len(dataset.files[i]) > class_max_size:
                class_max_size = len(dataset.files[i])
        num_steps = int(class_max_size / (self.param['batch_size']/len(dataset.files)))
        if class_max_size%(self.param['batch_size']/len(dataset.files)) != 0:
            num_steps += 1
        print("Num steps:", num_steps, "Data size:", data_size)
        start_time = time.time()
        for step in range(num_steps):
            patients, image_data, label_data = dataset.next_batch()
            if len(image_data) != self.param['ensemble_count']:
                # Ensembling over epochs. Use same data
                image_data = [image_data for i in range(0, self.param[
                                'ensemble_count'])]
            logits_ = []
            for m in range(0, self.param['ensemble_count']):
                print("Mode:", self.mode[m])
                layer_path = 'Train' + self.mode[m] + '/' + self.mode[m] + \
                             'logits/' + self.mode[m] + 'logits:0'
                logits_.append(self.cnnutils.get_features(self.mode[m],
                                                          self.meta_paths[m],
                                                          image_data[m],
                                                          layer_path))
            logits = np.average(logits_, axis=0)
            correct_ = np.equal(np.argmax(label_data, 1),
                                  np.argmax(logits, 1))
            answer = np.argmax(logits, 1)
            prediction_data.update(dict(zip(patients, answer)))
            pred_out.update(dict(zip(patients, correct_)))
            correct_predictions += np.sum(correct_.astype(float))
            total_seen += self.param['batch_size']
            print("Accuracy until " + str(total_seen) + " data points is: " +
                  str(correct_predictions / total_seen))
            #print("logits:", logits_)
        time_taken = time.time() - start_time
        accuracy_ = 0
        for key, value in pred_out.items():
            if value == True:
                accuracy_ += 1
        accuracy_ /= len(pred_out)
        print("Accuracy of ", len(pred_out)," images is ", accuracy_, "Time", time_taken)
        sys.stdout.flush()
        with open('./prediction_output.pkl', 'wb') as filep:
            pickle.dump(prediction_data, filep)
        csvwriter = csv.writer(open('./prediction_output.csv', 'w'))
        for key, value in prediction_data.items():
            csvwriter.writerow([key, value])
        #print(prediction_data, flush=True)
        return accuracy_

    def train(self, train_data, validation_data, flag):
        """
        This function creates the training operations and starts building and
        training the 3D CNN model.

        Args:
            validation_data: validation data to test the accuracy of the model.
        """
        with tf.Graph().as_default():
            print("Ensembling models..")
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.InteractiveSession(config=config)
            sess.run(init)
            print("Training Accuracy:",  self.evaluation(
                dataset=train_data))
            print("Validation Accuracy:", self.evaluation(
                dataset=validation_data))

