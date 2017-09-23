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

class CNNEnsemble:
    """
    This class provides functions to train a 3D Convolutional Neural Network
    model. To train the network and evaluate it, initialize the class with the
    required parameter file and call the function train() with training and
    validation datasets.
    """

    def __init__(self, params):
        self.param = params['cnn']
        self.modalities = 'ADNI_T1'
        self.model_paths = params['cnn']['models']
        self.checkpoint = params['cnn']['checkpoint']

    @classmethod
    def inference_loss(cls, logits, labels):
        """
        This function calculates the cross entropy loss from the output of the
        3D CNN model
        Args:
            logits: the output of 3D CNN [batch_size, 2]
            labels: the actual class labels of the batch [batch_size, 2]

        Returns: cross entropy loss

        """

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_loss')
        tf.summary.tensor_summary('logits', logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='batch_cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')




    def get_features(self, model, image_data, label_data):
        with tf.Graph().as_default() as model_graph:
            sess = tf.Session(graph=model_graph)
            ckpt = tf.train.get_checkpoint_state(self.checkpoint)
            #print(ckpath, ckpt)
            #print(model_graph.get_operations())
            if ckpt and ckpt.model_checkpoint_path:
                #sess.run(tf.initialize_all_variables())
                #print([tensor.name for tensor in model_graph.as_graph_def().node])
                model_path = self.checkpoint + self.model_paths[model]
                saver = tf.train.import_meta_graph(model_path+'.meta')
                saver.restore(sess, model_path)
                images = model_graph.get_tensor_by_name(self.modalities+'images:0')
                #images = model_graph.get_tensor_by_name('Placeholder:0')
                labels = model_graph.get_tensor_by_name(self.modalities+'labels:0')
                #labels = model_graph.get_tensor_by_name('Placeholder_1:0')
                keep_prob = model_graph.get_tensor_by_name(self.modalities+'keep_prob:0')
                #keep_prob = model_graph.get_tensor_by_name('Placeholder_3:0')
                #is_training = model_graph.get_tensor_by_name('phase:0')
                layer_ = []

                layer_path = 'Train'+self.modalities+'/'+self.modalities+'logits/'+self.modalities+'logits:0'
                layer = model_graph.get_tensor_by_name(layer_path)
                layer_ = sess.run(layer,
                                feed_dict={
                                    images: image_data,
                                    labels: label_data,
                                    keep_prob: 1.0
                                })
                print("Combining a layer with shape:", layer_.shape)
                return layer_
            else:
                print("No checkpoint found")
                return []

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
            logits_ = []
            for model in range(0, self.param['ensemble_epochs']):
                logits_.append(self.get_features(model, image_data,
                                                 label_data))
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
        # TODO: Add accuracy [2]
        sys.stdout.flush()
        with open('./prediction_output.pkl', 'wb') as filep:
            pickle.dump(prediction_data, filep)
        csvwriter = csv.writer(open('./prediction_output.csv', 'w'))
        for key, value in prediction_data.items():
            csvwriter.writerow([key, value])
        #print(prediction_data, flush=True)
        return accuracy_

    def train(self, train_data, validation_data, cad_data, test):
        """
        This function creates the training operations and starts building and
        training the 3D CNN model.

        Args:
            train_data: the training data required for 3D CNN
            validation_data: validation data to test the accuracy of the model.

        """
        with tf.Graph().as_default():


            #ckpt = tf.train.get_checkpoint_state(
            #    self.param['checkpoint_path'])
            #if ckpt and ckpt.model_checkpoint_path:
            #    saver.restore(sess, ckpt.model_checkpoint_path)
            print("Testing model")
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.InteractiveSession(config=config)
            sess.run(init)
            print("CAD:", self.evaluation(dataset=cad_data))
            print("ADNI+AIBL:", self.evaluation(dataset=validation_data))

