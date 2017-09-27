"""
This module contains the helper functions for building and training the 3D CNN
architecture.
"""

from datetime import datetime
import time
import math
import tensorflow as tf
import numpy as np
import sys


class CNNUtils:
    """
    Initialize the class with the required parameter file.
    """

    def __init__(self, params):
        self.param = params['cnn']

    @classmethod
    def variable_on_gpu(cls, name, shape, initializer):
        """
        This helper functions creates all the trainable weights on the GPU
        Args:
            name: name of the weights
            shape: the shape for these weights
            initializer: initializing these weights using tf initializers

        Returns: the initialized weights created with given shape

        """
        with tf.device('/gpu:0'):
            var = tf.get_variable(name=name,
                                  shape=shape,
                                  initializer=initializer)
        return var

    def weight_decay_variable(self, name, shape):
        """
        This function creates weights and adds a L2 regularisation loss
        Args:
            name: name of the weights
            shape: shape of these weights
            stddev: this is the standard deviation for initializing the 
                    weights. If we initialize the weights to a unit standard
                    deviation, the variance of the outputs of neuron increases
                    linearly with inputs. Hence, at present, the weights
                    are normalized by scaling it with square root of number of
                    inputs.
            Reference: http://cs231n.github.io/neural-networks-2/#init
        Returns: the initialized weights with weight decay

        """
        """
        decay_constant: Weight decay constant decides how much to penalize
                            the L2 regularization loss value. For big training
                            data, keep this value at minimum. For deeper
                            networks, keep a little higher value.
                            L1 regularization term is > L2 , so keep L1 wd < L2
        """
        temp_shape = shape[:-1]
        input_size = np.prod(temp_shape)
        stddev = 1.0 / math.sqrt(input_size / 2)
        weights = self.variable_on_gpu(name,
                                       shape,
                                       tf.random_normal_initializer(
                                           mean=0.0, stddev=stddev,
                                           seed=1)
                                       )
        weight_decay = tf.multiply(tf.nn.l2_loss(weights),
                                   self.param['decay_const'],
                                   name='weight_loss')

        tf.add_to_collection('losses', weight_decay)
        tf.add_to_collection('l2loss', weight_decay)
        return weights

    def conv_relu(self, input_, kernel_shape, biases_shape, scope):
        """
        This function builds a convolution layer of the 3D CNN
        Args:
            input_: input of the CNN layer maybe an image or an intermediate
                    feature representation
            kernel_shape: the shape of the kernel filters
            biases_shape: bias shape is equal to the shape of output channels
            scope: scope of the weights and biases in this convolution layer
        Returns:
            Feature maps of the convolution layer
        """

        weights = self.weight_decay_variable("weights", kernel_shape)
        tf.summary.histogram('weights', weights)
        biases = self.variable_on_gpu(name="biases",
                                      shape=biases_shape,
                                      initializer=tf.constant_initializer(
                                          0.001))
        tf.summary.histogram('biases', biases)

        conv = tf.nn.conv3d(input=input_,
                            filter=weights,
                            strides=[1, 2, 2, 2, 1],
                            padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        act_relu = tf.nn.relu(features=pre_activation, name=scope.name)
        return act_relu

    @classmethod
    def inference_loss(cls, logits, labels):
        """
        This function calculates the average cross entropy loss of the input
        batch and adds it to the 'loss' collections
        Args:
            logits: the output of 3D CNN
            labels: the actual class labels of the batch
        Returns: 

        """

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_loss')
        tf.summary.tensor_summary('logits', logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='batch_cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.add_to_collection('crossloss', cross_entropy_mean)

    def inference_conv2(self, conv1, keep_prob):
        with tf.variable_scope(self.param['mode']+'conv2') as scope:
            conv2 = self.conv_relu(input_=conv1,
                                       kernel_shape=[5, 5, 5, 45, 60],
                                       biases_shape=[60], scope=scope)
        print("Conv2", conv2.get_shape())
        with tf.variable_scope(self.param['mode']+'conv3') as scope:
            conv3 = self.conv_relu(input_=conv2,
                                   kernel_shape=[5, 5, 5, 60, 64],
                                   biases_shape=[64], scope=scope)
        print("Conv3", conv3.get_shape())
        with tf.variable_scope(self.param['mode']+'conv4') as scope:
            conv4 = self.conv_relu(input_=conv3,
                                   kernel_shape=[3, 3, 3, 64, 100],
                                   biases_shape=[100], scope=scope)
        print("Conv4", conv4.get_shape())

        with tf.variable_scope(self.param['mode']+'conv5') as scope:
            conv5 = self.conv_relu(input_=conv4,
                                   kernel_shape=[3, 3, 3, 100, 128],
                                   biases_shape=[128], scope=scope)
        print("Conv5", conv5.get_shape())
        with tf.variable_scope(self.param['mode']+'conv6') as scope:
            conv6 = self.conv_relu(input_=conv5,
                                   kernel_shape=[3, 3, 3, 128, 256],
                                   biases_shape=[256], scope=scope)
        print("Conv6", conv6.get_shape())
        with tf.variable_scope(self.param['mode']+'conv7') as scope:
            conv7 = self.conv_relu(input_=conv6,
                                   kernel_shape=[3, 3, 3, 256, 512],
                                   biases_shape=[512], scope=scope)
        print("Conv7", conv7.get_shape())

        with tf.variable_scope(self.param['mode']+'fullcn') as scope:
            vector_per_batch = tf.reshape(conv7, [self.param['batch_size'],
                                          -1])
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[512, 512])
            biases = self.variable_on_gpu(name="biases",
                                          shape=[512],
                                          initializer=tf.constant_initializer(
                                              0.01))
            pre_activation = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope(self.param['mode']+'logits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[512, self.param[
                                                     'classes']])
            biases = self.variable_on_gpu(name='biases',
                                          shape=[self.param['classes']],
                                          initializer=tf.constant_initializer(
                                              0.01))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)
            print('logits:', logits.get_shape())

        return logits

    def num_steps(self, dataset):
        """
        This function returns the number of steps to run to iterate the dataset
        Args:
            dataset: The dataset used for training/testing
        Returns: number of steps to run
        """
        class_max_size = 0
        data_size = 0
        for i in range(0, len(dataset.files)):
            data_size += len(dataset.files[i])
            if len(dataset.files[i]) > class_max_size:
                class_max_size = len(dataset.files[i])
        num_steps = int(
            class_max_size / (self.param['batch_size'] / len(dataset.files)))
        if class_max_size % (
                    self.param['batch_size'] / len(dataset.files)) != 0:
            num_steps += 1
        print("Num steps:", num_steps, "Data size:", data_size, flush=True)
        return num_steps

    def get_features(self, mode, meta_path, image_data, layer_path):
        with tf.Graph().as_default() as model_graph:
            sess = tf.Session(graph=model_graph)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, meta_path.split('.meta')[0])
            images = model_graph.get_tensor_by_name(mode + 'images:0')
            keep_prob = model_graph.get_tensor_by_name(mode + 'keep_prob:0')
            if layer_path == 'conv1':
                fusion_layer = 'conv1_a'
                layer_path_a = 'Train' + mode + '/' + mode + fusion_layer +\
                               '/' + mode + fusion_layer + ':0'
                fusion_layer = 'conv1_b'
                layer_path_b = 'Train' + mode + '/' + mode + fusion_layer +\
                               '/' + mode + fusion_layer + ':0'
                fusion_layer = 'conv1_c'
                layer_path_c = 'Train' + mode + '/' + mode + fusion_layer + \
                               '/' + mode + fusion_layer + ':0'
                layer_a = model_graph.get_tensor_by_name(layer_path_a)
                layer_b = model_graph.get_tensor_by_name(layer_path_b)
                layer_c = model_graph.get_tensor_by_name(layer_path_c)
                layer_a_, layer_b_, layer_c_ = sess.run(
                    [layer_a, layer_b, layer_c],
                    feed_dict={
                        images: image_data,
                        keep_prob: 1.0
                    })
                layer_ = np.concatenate([layer_a_, layer_b_, layer_c_], 4)
                print("Output layer shape:", layer_.shape)
                return layer_
            else:
                layer = model_graph.get_tensor_by_name(layer_path)
                layer_ = sess.run(layer,
                                  feed_dict={
                                      images: image_data,
                                      keep_prob: 1.0
                                  })
                print("Output layer shape:", layer_.shape)
                return layer_

    def fusion_conv1(self, fusion_input, keep_prob):
        prefix = 'Fusion'
        with tf.variable_scope(prefix + 'conv2') as scope:
            conv2 = self.conv_relu(input_=fusion_input,
                                            kernel_shape=[5, 5, 5, 135, 135],
                                            biases_shape=[135], scope=scope)
        print("Conv2", conv2.get_shape())
        with tf.variable_scope(prefix + 'conv3') as scope:
            conv3 = self.conv_relu(input_=conv2,
                                            kernel_shape=[5, 5, 5, 135, 140],
                                            biases_shape=[140], scope=scope)
        print("Conv3", conv3.get_shape())
        with tf.variable_scope(prefix + 'conv4') as scope:
            conv4 = self.conv_relu(input_=conv3,
                                            kernel_shape=[3, 3, 3, 140, 145],
                                            biases_shape=[145], scope=scope)
        print("Conv4", conv4.get_shape())
        with tf.variable_scope(prefix + 'conv5') as scope:
            conv5 = self.conv_relu(input_=conv4,
                                            kernel_shape=[3, 3, 3, 145, 150],
                                            biases_shape=[150], scope=scope)
        print("Conv5", conv5.get_shape())
        with tf.variable_scope(prefix + 'conv6') as scope:
            conv6 = self.conv_relu(input_=conv5,
                                            kernel_shape=[3, 3, 3, 150, 256],
                                            biases_shape=[256], scope=scope)
        print("Conv6", conv6.get_shape())
        with tf.variable_scope(prefix + 'conv7') as scope:
            conv7 = self.conv_relu(input_=conv6,
                                            kernel_shape=[3, 3, 3, 256, 512],
                                            biases_shape=[512], scope=scope)
        print("Conv7", conv7.get_shape())
        with tf.variable_scope(prefix + 'fullcn') as scope:
            vector_per_batch = tf.reshape(conv7, [self.param['batch_size'],
                                                  -1])
            weights = self.weight_decay_variable(name="weights",
                                                          shape=[512, 512])
            biases = self.variable_on_gpu(name="biases",
                                                   shape=[512],
                                                   initializer=tf.constant_initializer(
                                                       0.01))
            pre_activation = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope(prefix + 'logits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                          shape=[512,
                                                                 self.param[
                                                                     'classes']])
            biases = self.variable_on_gpu(name='biases',
                                                   shape=[
                                                       self.param['classes']],
                                                   initializer=tf.constant_initializer(
                                                       0.01))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)
            print('logits:', logits.get_shape())

        return logits

    def fusion_conv7(self, fusion_input, keep_prob):
        with tf.variable_scope('Fusionfullcn') as scope:
            vector_per_batch = tf.reshape(fusion_input, [self.param['batch_size'],
                                                  -1])
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[1536, 512])
            biases = self.variable_on_gpu(name="biases",
                                          shape=[512],
                                          initializer=tf.constant_initializer(
                                              0.01))
            pre_activation = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope('Fusionlogits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[512, self.param[
                                                     'classes']])
            biases = self.variable_on_gpu(name='biases',
                                          shape=[self.param['classes']],
                                          initializer=tf.constant_initializer(
                                              0.01))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)
            print('logits:', logits.get_shape())

        return logits

    def evaluation(self, sess, eval_op, dataset, images, labels, keep_prob,
                   loss, corr, xloss, l2loss):
        """
        This function evaluates the accuracy of the model
        Args:
            sess: tensorflow session
            eval_op: evaluation operation which calculates number of correct
                    predictions
            dataset: input dataset
            images: the images placeholder
            labels: the labels placeholder
            keep_prob: dropout keep probability placeholder
            loss: total loss tensor
            corr: correct predictions tensor
            xloss: cross entropy loss tensor
            l2loss: L2 loss tensor

        Returns: the accuracy of the 3D CNN model

        """
        correct_predictions = 0
        total_seen = 0
        pred_out = {}
        num_steps = self.num_steps(dataset)
        for step in range(num_steps):
            patients, image_data, label_data = dataset.next_batch()
            predictions, correct_, loss_, xloss_, l2loss_ = sess.run(
                [eval_op, corr, loss, xloss, l2loss],
                feed_dict={
                    images: image_data,
                    labels: label_data,
                    keep_prob: 1.0
                })
            correct_predictions += predictions
            pred_out.update(dict(zip(patients, correct_)))
            total_seen += self.param['batch_size']
            print("Accuracy until " + str(total_seen) + " data points is: " +
                  str(correct_predictions / total_seen))
            print("loss", loss_, xloss_, l2loss_)
        accuracy_ = 0
        for key, value in pred_out.items():
            if value == True:
                accuracy_ += 1
        accuracy_ /= len(pred_out)
        print("Accuracy of ", len(pred_out), " images is ", accuracy_,
              flush=True)
        return accuracy_
