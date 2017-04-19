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
        weight_decay = tf.multiply(tf.nn.l2_loss(weights),
                                   decay_constant,
                                   name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return weights

    def conv_relu(self, input_, kernel_shape, biases_shape, decay_constant,
                  scope):
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
                            strides=[1, 1, 1, 1, 1],
                            padding='VALID')
        pre_activation = tf.nn.bias_add(conv, biases)

        return tf.nn.relu(features=pre_activation, name=scope.name)

    def inference(self, images):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print(images.get_shape())
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_relu(input_=images,
                                   kernel_shape=[5, 5, 5, 1, 8],
                                   biases_shape=[8],
                                   decay_constant=0.001,
                                   scope=scope)

        pool1 = tf.nn.max_pool3d(conv1,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")

        with tf.variable_scope('conv2') as scope:
            conv2 = self.conv_relu(input_=pool1,
                                   kernel_shape=[5, 5, 5, 8, 32],
                                   biases_shape=[32],
                                   decay_constant=0.004,
                                   scope=scope)

        pool2 = tf.nn.max_pool3d(conv2,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")

        with tf.variable_scope('conv3') as scope:
            conv3 = self.conv_relu(input_=pool2,
                                   kernel_shape=[5, 5, 5, 32, 64],
                                   biases_shape=[64],
                                   decay_constant=0.004,
                                   scope=scope)

        pool3 = tf.nn.max_pool3d(conv3,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")

        with tf.variable_scope('conv4') as scope:
            conv4 = self.conv_relu(input_=pool3,
                                   kernel_shape=[5, 5, 5, 64, 64],
                                   biases_shape=[64],
                                   decay_constant=0.004,
                                   scope=scope)

        pool4 = tf.nn.max_pool3d(conv4,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")

        with tf.variable_scope('conv5') as scope:
            conv5 = self.conv_relu(input_=pool4,
                                   kernel_shape=[5, 5, 5, 64, 128],
                                   biases_shape=[128],
                                   decay_constant=0.004,
                                   scope=scope)

        pool5 = tf.nn.max_pool3d(conv5,
                                 ksize=[1, 2, 2, 2, 1],
                                 strides=[1, 2, 2, 2, 1],
                                 padding="SAME")

        with tf.variable_scope('fullcn') as scope:
            vector_per_batch = tf.reshape(pool5,
                                          [self.param['batch_size'], -1])
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[2304, 1024],
                                                 decay_constant=0.004)
            biases = self.variable_on_cpu(name="biases",
                                          shape=[1024],
                                          initializer=tf.constant_initializer(
                                              0.1))
            fullcn = tf.nn.relu(tf.matmul(vector_per_batch, weights) + biases,
                                name=scope.name)

        with tf.variable_scope('fullcn2') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[1024, 64],
                                                 decay_constant=0.004)
            biases = self.variable_on_cpu(name="biases",
                                          shape=[64],
                                          initializer=tf.constant_initializer(
                                              0.1))
            fullcn2 = tf.nn.relu(tf.matmul(fullcn, weights) + biases,
                                 name=scope.name)

        with tf.variable_scope('logits') as scope:
            weights = self.weight_decay_variable(name="weights",
                                                 shape=[64, 2],
                                                 decay_constant=0.001)
            biases = self.variable_on_cpu(name='biases',
                                          shape=[2],
                                          initializer=tf.constant_initializer(
                                              0.1))
            logits = tf.add(tf.matmul(fullcn2, weights), biases,
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
        tf.summary.tensor_summary('logits', logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='batch_cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def evaluation(self, eval_op, dataset, images, labels):
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
            predictions = eval_op.eval(
                feed_dict={
                    images: image_data,
                    labels: label_data
                })
            correct_predictions += predictions
            if step % 9 == 0:
                total_seen = (step + 1) * self.param['batch_size']
                print("Accuracy until "+str(total_seen)+" data points is: " +
                      str(correct_predictions/total_seen))

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
                                    shape=[None,
                                           self.param['depth'],
                                           self.param['height'],
                                           self.param['width'],
                                           1])
            labels = tf.placeholder(dtype=tf.int8,
                                    shape=[None, 2])

            global_step = tf.get_variable(name='global_step',
                                          shape=[],
                                          initializer=tf.constant_initializer(
                                              0),
                                          trainable=False)
            train_size = len(train_data.filenames)
            num_batches = int(train_size / self.param['batch_size'])
            num_steps = num_batches * self.param['num_epochs']

            learn_rate = tf.train.exponential_decay(
                self.param['learning_rate'], global_step,
                decay_steps=num_steps, decay_rate=self.param['decay_factor'],
                staircase=True)
            opt = tf.train.AdamOptimizer(
                learning_rate=self.param['learning_rate'])

            tf.summary.scalar('learning_rate', learn_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train') as scope:
                    logits = self.inference(images)

                    _ = self.inference_loss(logits, labels)

                    losses = tf.get_collection('losses', scope)

                    # Sum all the losses
                    total_loss = tf.add_n(losses, name='total_loss')
                    tf.summary.scalar('total_loss', total_loss)

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
                sess = tf.InteractiveSession()
                sess.run(init)

                # Create a summary writer
                summary_writer = tf.summary.FileWriter(
                    self.param['summary_path'], sess.graph)
                summary_op = tf.summary.merge_all()
                tf.get_default_graph().finalize()

                for step in range(1, num_steps):
                    start_time = time.time()

                    image_data, label_data = train_data.next_batch()
                    summary_values, _, loss_value = sess.run(
                        [summary_op,
                         train_op,
                         total_loss],
                        feed_dict={
                            images: image_data,
                            labels: label_data
                        }
                    )
                    if step % 10 == 0:
                        accuracy_ = sess.run(accuracy,
                                             feed_dict={
                                                 images: image_data,
                                                 labels: label_data})
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
                    summary_writer.add_summary(summary_values, step)

                    # Saving Model Checkpoints for evaluation
                    if step % num_batches == 0 or (step + 1) == num_steps:
                        checkpoint_path = self.param['checkpoint_path'] + \
                                          '/model.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)

                        # Evaluate against the training data.
                        print("Step: %d Training accuracy: %g " %
                              (step, self.evaluation(eval_op=eval_op,
                                                     dataset=train_data,
                                                     images=images,
                                                     labels=labels)))

                        # Evaluate against the validation data
                        print("Step: %d Validation accuracy: %g" %
                              (step, self.evaluation(eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=images,
                                                     labels=labels)))
