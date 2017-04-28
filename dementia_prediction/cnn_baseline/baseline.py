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
        This helper functions creates all the weights on the CPU
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
            stddev: this is the standard deviation for intialising the weights.
                    If we initialize the weights to a unit standard
                    deviation, the variance of the outputs of neuron increases
                    linearly with inputs. Hence, at present, the weights
                    are normalized by scaling it with square root of number of
                    inputs.
            Reference: http://cs231n.github.io/neural-networks-2/#init

        Returns: the initialized weights with weight decay

        """

        temp_shape = shape[:-1]
        input_size = np.prod(temp_shape)
        stddev = 1.0 / math.sqrt(input_size / 2)
        weights = self.variable_on_cpu(name,
                                       shape,
                                       tf.random_normal_initializer(
                                           mean=0.0, stddev=stddev,
                                           seed=1))
        if decay_constant > 0:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights),
                                   decay_constant,
                                   name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return weights

    def conv_relu(self, input_, kernel_shape, biases_shape, decay_constant,
                  scope, padding, stride, is_training):
        """
        This function builds a convolution layer of the 3D CNN
        Args:
            input_: input of the CNN layer maybe an image or an intermediate
                    feature representation
            kernel_shape: the shape of the kernel filters
            biases_shape: bias shape is equal to the shape of output channels
            wd: Weight decay adds weight to the L2 regularization loss value.
                For more training data, keep this value at minimum
                For deeper networks, keep a little higher value
                L1 regularization term is > L2 , so keep L1 wd < L2
            scope: scope of the weights and biases in this convolution layer

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
        bn1 = tf.contrib.layers.batch_norm(act_relu,
                                           center=True, scale=True,
                                           is_training=is_training)
        return bn1

    def inference(self, images, keep_prob, is_training):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_relu(input_=images,
                                   kernel_shape=[7, 7, 7, 1, 10],
                                   biases_shape=[10],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv1)
        """
        pool1 = tf.nn.max_pool3d(conv1,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv_pool1') as scope:
            conv_pool1 = self.conv_relu(input_=conv1,
                                   kernel_shape=[2, 2, 2, 10, 10],
                                   biases_shape=[10],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("conv_pool1:",conv_pool1.get_shape())
        with tf.variable_scope('conv2') as scope:
            conv2 = self.conv_relu(input_=conv_pool1,
                                   kernel_shape=[6, 6, 6, 10, 32],
                                   biases_shape=[32],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv2)
        with tf.variable_scope('conv_pool2') as scope:
            conv_pool2 = self.conv_relu(input_=conv2,
                                   kernel_shape=[2, 2, 2, 32, 32],
                                   biases_shape=[32],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("conv_pool2:",conv_pool2.get_shape())
        """
        pool2 = tf.nn.max_pool3d(conv2,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv3') as scope:
            conv3 = self.conv_relu(input_=conv_pool2,
                                   kernel_shape=[5, 5, 5, 32, 64],
                                   biases_shape=[64],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv3)
        with tf.variable_scope('conv_pool3') as scope:
            conv_pool3 = self.conv_relu(input_=conv3,
                                   kernel_shape=[2, 2, 2, 64, 64],
                                   biases_shape=[64],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("conv_pool3:",conv_pool3.get_shape())
        """
        pool3 = tf.nn.max_pool3d(conv3,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv4') as scope:
            conv4 = self.conv_relu(input_=conv_pool3,
                                   kernel_shape=[3, 3, 3, 64, 100],
                                   biases_shape=[100],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv4)

        with tf.variable_scope('conv_pool4') as scope:
            conv_pool4 = self.conv_relu(input_=conv4,
                                   kernel_shape=[2, 2, 2, 100, 100],
                                   biases_shape=[100],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("conv_pool4:",conv_pool4.get_shape())
        """
        pool4 = tf.nn.max_pool3d(conv4,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv5') as scope:
            conv5 = self.conv_relu(input_=conv_pool4,
                                   kernel_shape=[3, 3, 3, 100, 128],
                                   biases_shape=[128],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv5)

        with tf.variable_scope('conv_pool5') as scope:
            conv_pool5 = self.conv_relu(input_=conv5,
                                   kernel_shape=[2, 2, 2, 128, 128],
                                   biases_shape=[128],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("convpool5:",conv_pool5.get_shape())
        """
        pool5 = tf.nn.max_pool3d(conv5,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv6') as scope:
            conv6 = self.conv_relu(input_=conv_pool5,
                                   kernel_shape=[3, 3, 3, 128, 200],
                                   biases_shape=[200],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv6)
        with tf.variable_scope('conv_pool6') as scope:
            conv_pool6 = self.conv_relu(input_=conv6,
                                   kernel_shape=[2, 2, 2, 200, 200],
                                   biases_shape=[200],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("convpool6:",conv_pool6.get_shape())
        """
        pool6 = tf.nn.max_pool3d(conv6,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        with tf.variable_scope('conv7') as scope:
            conv7 = self.conv_relu(input_=conv_pool6,
                                   kernel_shape=[3, 3, 3, 200, 256],
                                   biases_shape=[256],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv7)
        """
        pool7 = tf.nn.max_pool3d(conv7,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")
        """
        print("conv7",conv7.get_shape())
        """
        with tf.variable_scope('conv8') as scope:
            conv8 = self.conv_relu(input_=conv7,
                                   kernel_shape=[5, 5, 5, 256, 512],
                                   biases_shape=[512],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=1,
                                   is_training=is_training)
            tf.summary.histogram('activations',conv8)
        with tf.variable_scope('conv_pool8') as scope:
            conv_pool8 = self.conv_relu(input_=conv8,
                                   kernel_shape=[2, 2, 2, 512, 512],
                                   biases_shape=[512],
                                   decay_constant=self.param['decay_const'],
                                   scope=scope,
                                   padding='SAME', stride=2,
                                   is_training=is_training)
            print("convpool8",conv_pool8.get_shape())
        """
        with tf.variable_scope('fullcn') as scope:
            vector_per_batch = tf.reshape(conv7,
                                          [self.param['batch_size'], -1])
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[2048, 1024],
                                   decay_constant=0)
            biases = self.variable_on_cpu(name="biases",
                                          shape=[1024],
                                          initializer=tf.constant_initializer(
                                              0.1))
            fullcn = tf.nn.relu(tf.matmul(vector_per_batch, weights) + biases,
                                name=scope.name)
            h2 = tf.contrib.layers.batch_norm(fullcn,
                                              center=True, scale=True,
                                              is_training=is_training)
            # Use dropout if network overfits
            # fullcn_drop = tf.nn.dropout(h2, keep_prob)

        with tf.variable_scope('fullcn2') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[1024, 64],
                                   decay_constant=0)
            biases = self.variable_on_cpu(name="biases",
                                          shape=[64],
                                          initializer=tf.constant_initializer(
                                              0.1))
            fullcn2 = tf.nn.relu(tf.matmul(h2, weights) + biases,
                                 name=scope.name)
            h3 = tf.contrib.layers.batch_norm(fullcn2,
                                              center=True, scale=True,
                                              is_training=is_training)
            # fullcn2_drop = tf.nn.dropout(h3, keep_prob)

        with tf.variable_scope('logits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[64, 2],
                                   decay_constant=0)
            biases = self.variable_on_cpu(name='biases',
                                          shape=[2],
                                          initializer=tf.constant_initializer(
                                              0.1))
            logits = tf.add(tf.matmul(h3, weights), biases,
                            name=scope.name)

        return logits

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
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='batch_cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def evaluation(self, sess, eval_op, dataset, images, labels, keep_prob,
                   is_training, loss):
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
        dataset_size = len(dataset.filenames)
        for step in range(int(dataset_size / self.param['batch_size'])):
            image_data, label_data = dataset.next_batch()
            predictions, loss_ = sess.run([eval_op, loss],
                feed_dict={
                    images: image_data,
                    labels: label_data,
                    keep_prob: 1.0,
                    is_training: 0
                })
            correct_predictions += predictions
            if step % 9 == 0:
                total_seen = (step + 1) * self.param['batch_size']
                print("Accuracy until "+str(total_seen)+" data points is: " +
                      str(correct_predictions/total_seen))
                print("loss", loss_)

        accuracy = correct_predictions / dataset_size
        return accuracy

    def train(self, train_data, validation_data):
        """
        This function creates the training operations and starts building and
        training the 3D CNN model.

        Args:
            train_data: the training data required for 3D CNN
            validation_data: validation data to test the accuracy of the model.

        """
        with tf.Graph().as_default():

            images = tf.placeholder(dtype=tf.float32,
                                    shape=[None,self.param['depth'],
                                         self.param['height'],
                                         self.param['width'],
                                           1])
            labels = tf.placeholder(dtype=tf.int8,
                                    shape=[None, 2])
            keep_prob = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool, name='phase')
            global_step = tf.get_variable(name='global_step',
                                          shape=[],
                                          initializer=tf.constant_initializer(
                                              0),
                                          trainable=False)
            train_size = len(train_data.filenames)
            num_batches_epoch = int(train_size / self.param['batch_size'])
            num_steps = num_batches_epoch * self.param['num_epochs']

            learn_rate = tf.train.exponential_decay(
                self.param['learning_rate'], global_step,
                decay_steps=num_steps, decay_rate=self.param['decay_factor'],
                staircase=True)
            opt = tf.train.AdamOptimizer(
                learning_rate=self.param['learning_rate'])

            # tf.summary.scalar('learning_rate', learn_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train') as scope:
                    logits = self.inference(images, keep_prob, is_training)
                    #tf.summary.tensor_summary('logits', logits)

                    _ = self.inference_loss(logits, labels)

                    losses = tf.get_collection('losses', scope)

                    # Sum all the losses
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)
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

                for step in range(1, num_steps):
                    start_time = time.time()

                    image_data, label_data = train_data.next_batch()
                    summary_values, _, loss_value, logits_ = sess.run(
                        [summary_op,
                         train_op,
                         total_loss,
                         logits],
                        feed_dict={
                            images: image_data,
                            labels: label_data,
                            keep_prob: self.param['keep_prob'],
                            is_training: 1
                        }
                    )
                    if step % 10 == 0:
                        accuracy_ = sess.run(accuracy,
                                             feed_dict={
                                                 images: image_data,
                                                 labels: label_data,
                                                 keep_prob: 1.0,
                                                 is_training: 0})
                        print("Train Batch Accuracy. %g step %d" % (accuracy_,
                                                                    step))
                        summary_writer.add_summary(summary_values, step)
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
                        #print(logits_)

                    # Saving Model Checkpoints
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps:
                        # After 15 epochs or before the last step
                        if (step/num_batches_epoch) % 15 == 0 or (step+1) == num_steps:
                            checkpoint_path = self.param['checkpoint_path'] + \
                                              '/model.ckpt'
                            saver.save(sess, checkpoint_path, global_step=step)

                        # Evaluate against the training data.
                        print("Evaluating on Training data...")
                        print("Step: %d Training accuracy: %g " %
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=train_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     is_training=is_training,
                                                     loss=total_loss)))

                        # Evaluate against the validation data
                        print("Evaluating on Validation data...")
                        print("Step: %d Validation accuracy: %g" %
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     is_training=is_training,
                                                     loss=total_loss)))
                    sys.stdout.flush()
