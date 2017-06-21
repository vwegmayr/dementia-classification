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

class CNN:
    """
    This class provides functions to train a 3D Convolutional Neural Network
    model. To train the network and evaluate it, initialize the class with the
    required parameter file and call the function train() with training and
    validation datasets.
    """

    def __init__(self, params):
        self.param = params['cnn']

    @classmethod
    def variable_on_cpu(cls, name, shape, initializer):
        """
        This helper functions creates all the trainable weights on the CPU
        Args:
            name: name of the weights
            shape: the shape for these weights
            initializer: initializing these weights using tf intializers

        Returns: the initialized weights created with given shape

        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name,
                                  shape=shape,
                                  initializer=initializer)
        return var

    def weight_decay_variable(self, name, shape, decay_constant):
        """
        This function creates weights and adds a L2 regularisation loss
        Args:
            name: name of the weights
            shape: shape of these weights
            decay_constant: Weight decay constant decides how much to penalize
                            the L2 regularization loss value. For big training
                            data, keep this value at minimum. For deeper
                            networks, keep a little higher value.
                            L1 regularization term is > L2 , so keep L1 wd < L2
            Comment on stddev:
                    If we initialize the weights to a unit standard
                    deviation, the variance of the outputs of neuron increases
                    linearly with inputs. Hence, at present, the weights
                    are normalized by scaling it with square root of number of
                    inputs.
            Reference: http://cs231n.github.io/neural-networks-2/#init

        Returns: the initialized weights with weight decay

        """

        """
        """
        temp_shape = shape[:-1]
        input_size = np.prod(temp_shape)
        stddev = 1.0 / math.sqrt(input_size / 2)
        weights = self.variable_on_cpu(name,
                                       shape,
                                       tf.random_normal_initializer(
                                            mean=0.0, stddev=stddev,
                                            seed=1)
                                       )
        #tf.contrib.layers.variance_scaling_initializer(mode='FAN_IN'))
        weight_decay = tf.multiply(tf.nn.l2_loss(weights),
                                   decay_constant,
                                   name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return weights

    def conv_relu(self, input_, kernel_shape, biases_shape, decay_constant,
                  scope, padding, stride):
        """
        This function builds a convolution layer of the 3D CNN
        Args:
            input_: input of the CNN layer maybe an image or an intermediate
                    feature representation
            kernel_shape: the shape of the kernel filters
            biases_shape: bias shape is equal to the shape of output channels
            decay_constant: 
                Weight decay adds weight to the L2 regularization loss value.
                For more training data, keep this value at minimum.
                For deeper networks, keep a little higher value. L1
                regularization term is > L2 , so keep L1 wd < L2
            scope: scope of the weights and biases in this convolution layer
            padding: 'SAME' or 'VALID' padding
            stride: integer, convolution stride

        Returns:
            Feature maps of the convolution layer

        """

        weights = self.weight_decay_variable("weights", kernel_shape,
                                             decay_constant)
        tf.summary.histogram('weights', weights)
        biases = self.variable_on_cpu(name="biases",
                                      shape=biases_shape,
                                      initializer=tf.constant_initializer(
                                          0.01))
        tf.summary.histogram('biases', biases)

        conv = tf.nn.conv3d(input=input_,
                            filter=weights,
                            strides=[1, stride, stride, stride, 1],
                            padding=padding)
        pre_activation = tf.nn.bias_add(conv, biases)
        act_relu = tf.nn.relu(features=pre_activation, name=scope.name)
        return act_relu 

    def inference(self, images, keep_prob):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print(images.get_shape())
        # Change 7,7,7 to 5,5,5
        with tf.variable_scope('conv1_a') as scope:
            conv1_a = self.conv_relu(input_=images,
                                   kernel_shape=[5, 5, 5, 1, 10],
                                   biases_shape=[10],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv1_a", conv1_a.get_shape())
        with tf.variable_scope('conv1_b') as scope:
            conv1_b = self.conv_relu(input_=images,
                                   kernel_shape=[6, 6, 6, 1, 10],
                                   biases_shape=[10],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv1_b", conv1_b.get_shape())
        with tf.variable_scope('conv1_c') as scope:
            conv1_c = self.conv_relu(input_=images,
                                   kernel_shape=[7, 7, 7, 1, 10],
                                   biases_shape=[10],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv1_c", conv1_c.get_shape())
        conv1 = tf.concat([conv1_a, conv1_b, conv1_c], 4)
        with tf.variable_scope('conv2') as scope:
            conv2 = self.conv_relu(input_=conv1,
                                   kernel_shape=[5, 5, 5, 30, 32],
                                   biases_shape=[32],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv2", conv2.get_shape())

        with tf.variable_scope('conv3') as scope:
            conv3 = self.conv_relu(input_=conv2,
                                   kernel_shape=[5, 5, 5, 32, 64],
                                   biases_shape=[64],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv3", conv3.get_shape())

        with tf.variable_scope('conv4') as scope:
            conv4 = self.conv_relu(input_=conv3,
                                   kernel_shape=[3, 3, 3, 64, 64],
                                   biases_shape=[64],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv4", conv4.get_shape())

        with tf.variable_scope('conv5') as scope:
            conv5 = self.conv_relu(input_=conv4,
                                   kernel_shape=[3, 3, 3, 64, 128],
                                   biases_shape=[128],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv5", conv5.get_shape())

        with tf.variable_scope('conv6') as scope:
            conv6 = self.conv_relu(input_=conv5,
                                   kernel_shape=[3, 3, 3, 128, 256],
                                   biases_shape=[256],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv6", conv6.get_shape())

        with tf.variable_scope('conv7') as scope:
            conv7 = self.conv_relu(input_=conv6,
                                   kernel_shape=[3, 3, 3, 256, 512],
                                   biases_shape=[512],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,padding='SAME',
                                   stride=2)
        print("Conv7", conv7.get_shape())

        with tf.variable_scope('fullcn2') as scope:
            vector_per_batch = tf.reshape(conv7, [self.param['batch_size'],
                                          -1])
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[512, 512],
                                   decay_constant=self.param['decay_const'])
            biases = self.variable_on_cpu(name="biases",
                                          shape=[512],
                                          initializer=tf.constant_initializer(
                                              0.1))
            pre_activation2 = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation2, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)

        with tf.variable_scope('logits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[512, 2],
                                   decay_constant=self.param['decay_const'])
            biases = self.variable_on_cpu(name='biases',
                                          shape=[2],
                                          initializer=tf.constant_initializer(
                                              0.1))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)

        return logits

    @classmethod
    def inference_loss(cls, logits, labels):
        """
        This function calculates the cross entropy loss from the output of the
        3D CNN model
        Args:
            logits: the output of 3D CNN Ex: [batch_size, 2]
            labels: the actual class labels of the batch Ex: [batch_size, 2]

        Returns: cross entropy loss

        """

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_loss')
        tf.summary.tensor_summary('logits', logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='batch_cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
    def evaluation(self, sess, eval_op, dataset, images, labels, keep_prob,
                    loss, corr):
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
            loss: cross entropy loss tensor
            corr: correct predictions tensor

        Returns: the accuracy of the 3D CNN model

        """
        correct_predictions = 0
        total_seen = 0
        dataset_size = len(dataset.s_files) + len(dataset.p_files)
        for step in range(int(dataset_size / self.param['batch_size'])):
            image_data, label_data = dataset.next_batch()
            predictions, correct_, loss_ = sess.run([eval_op, corr, loss],
                feed_dict={
                    images: image_data,
                    labels: label_data,
                    keep_prob: 1.0
                })
            print("Prediction:", correct_)
            correct_predictions += predictions

            total_seen += self.param['batch_size']
            print("Accuracy until "+str(total_seen)+" data points is: " +
                      str(correct_predictions/total_seen))
            print("loss", loss_)
        accuracy = correct_predictions / total_seen
        return accuracy

    def train(self, train_data, validation_data, test):
        """
        This function starts building and
        training the 3D CNN model.

        Args:
            train_data: the training dataset object of class DataInput
            validation_data: validation dataset object of class DataInput
            test: Data object of DataInput class to test the model accuracy
        """
        with tf.Graph().as_default():

            images = tf.placeholder(dtype=tf.float32,
                                    shape=[None,
                                           self.param['depth'],
                                           self.param['height'],
                                           self.param['width'],
                                           1], name='images')
            labels = tf.placeholder(dtype=tf.int8,
                                    shape=[None, 2], name='labels')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            var_lr = tf.placeholder(tf.float32, name='var_lr')
            global_step = tf.get_variable(name='global_step',
                                          shape=[],
                                          initializer=tf.constant_initializer(
                                              0),
                                          trainable=False)
            train_size = len(train_data.s_files) + len(train_data.p_files)
            num_batches_epoch = int(train_size / self.param['batch_size'])
            num_steps = num_batches_epoch * self.param['num_epochs']

            learn_rate = tf.train.exponential_decay(
                self.param['learning_rate'], global_step,
                decay_steps=num_steps, decay_rate=self.param['decay_factor'],
                staircase=True)
            opt = tf.train.AdamOptimizer(
                learning_rate=self.param['learning_rate'])

            #tf.summary.scalar('learning_rate', learn_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train') as scope:
                    logits = self.inference(images, keep_prob)

                    _ = self.inference_loss(logits, labels)

                    losses = tf.get_collection('losses', scope)

                    # Sum all the losses
                    total_loss = tf.add_n(losses, name='total_loss')
                    #tf.summary.scalar('total_loss', total_loss)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = opt.minimize(total_loss, global_step=global_step)

                # Evaluation
                correct_prediction = tf.equal(tf.argmax(labels, 1),
                                              tf.argmax(logits, 1))
                eval_op = tf.reduce_sum(tf.cast(correct_prediction,
                                                tf.float32))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))
                tf.summary.scalar('train_batch_accuracy', accuracy)
                saver = tf.train.Saver()
                init = tf.global_variables_initializer()
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True
                sess = tf.InteractiveSession(config=config)
                sess.run(init)
                
                # Create a summary writer
                summary_writer = tf.summary.FileWriter(
                    self.param['summary_path'], sess.graph)
                summary_op = tf.summary.merge_all()
                tf.get_default_graph().finalize()

                # For var_lr - decreasing the learning rate after specific
                # number of epochs
                init_lr = self.param['learning_rate']

                for step in range(1, num_steps):
                    start_time = time.time()
                    if step%(5*num_batches_epoch) == 0:
                        init_lr /= 2
                    image_data, label_data = train_data.next_batch()
                    summary_values, _, loss_value = sess.run(
                        [summary_op,
                         train_op,
                         total_loss],
                        feed_dict={
                            images: image_data,
                            labels: label_data,
                            keep_prob: self.param['keep_prob']
                        }
                    )
                    accuracy_ = sess.run(accuracy,
                                         feed_dict={
                                                 images: image_data,
                                                 labels: label_data,
                                                 keep_prob: 1.0
                                                 })
                    print("Train Batch Accuracy. %g step %d" % (accuracy_,
                                                                    step))
                    duration = time.time() - start_time

                    assert not np.isnan(
                        loss_value), 'Model diverged with loss = NaN'

                    if step % 5 == 0:
                        num_examples_per_step = self.param['batch_size']
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration

                        format_str = ('%s: step %d, loss = %.2f (%.1f '
                                      'examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            examples_per_sec, sec_per_batch))
                    #summary_writer.add_summary(summary_values, step)

                    # Saving Model Checkpoints for evaluation
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps:
                        if (step+1) == num_steps:
                            checkpoint_path = self.param['checkpoint_path'] + \
                                              'model.ckpt'
                            saver.save(sess, checkpoint_path, global_step=step)

                        # Evaluate against the training data.
                        print("Step: %d Training accuracy: %g " %
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=train_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss)))

                        # Evaluate against the validation data
                        print("Step: %d Validation accuracy: %g" %
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss)))
                    sys.stdout.flush()
                '''
                if test == True:
                    ckpt = tf.train.get_checkpoint_state(self.param['checkpoint_path']) 
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print("Testing model")
                        print(self.evaluation(sess=sess,
                                                 eval_op=eval_op,
                                                 dataset=validation_data,
                                                 images=images,
                                                 labels=labels,
                                                 keep_prob=keep_prob,
                                                 corr=correct_prediction,
                                                 loss=total_loss))
                    else:
                        print("No checkpoint found.")
                '''

