"""
This module builds the Baseline 3D CNN architecture 
"""

from datetime import datetime
import time
import math
import tensorflow as tf
import numpy as np
import sys
from dementia_prediction.cnn_utils import CNNUtils
# TODO Add transfer learning finetuning
class CNNBaseline:
    """
    This class provides functions to train the baseline 3D Convolutional 
    Neural Network model. To train the network and evaluate it, initialize the 
    class with the required parameter file and call the function train() with 
    training and validation datasets.
    """

    def __init__(self, params):
        self.param = params['cnn']
        self.mlp = params['mlp']
        self.cnnutils = CNNUtils(params)
        print("Parameters:", params, flush=True)

    def mlp_inference(self, images, keep_prob):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print(images.get_shape())
        with tf.variable_scope(self.param['mode'] + 'fullcn') as scope:
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                          shape=[self.param[
                                                                     'num_features'],
                                                                 8000])
            biases = self.cnnutils.variable_on_gpu(name="biases",
                                                   shape=[8000],
                                                   initializer=tf.constant_initializer(
                                                       0.01))
            pre_activation = tf.matmul(images, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope(self.param['mode'] + 'logits') as scope:
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                          shape=[8000,
                                                                 self.param[
                                                                     'classes']])
            biases = self.cnnutils.variable_on_gpu(name='biases',
                                                   shape=[
                                                       self.param['classes']],
                                                   initializer=tf.constant_initializer(
                                                       0.01))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)
            print('logits:', logits.get_shape())

        return logits

    def inference(self, images, keep_prob):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print("Input shape:", images.get_shape())
        with tf.variable_scope(self.param['mode']+'conv1_a') as scope:
            conv1_a = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[5, 5, 5, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_a", conv1_a.get_shape())
        with tf.variable_scope(self.param['mode']+'conv1_b') as scope:
            conv1_b = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[6, 6, 6, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_b", conv1_b.get_shape())
        with tf.variable_scope(self.param['mode']+'conv1_c') as scope:
            conv1_c = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[7, 7, 7, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_c", conv1_c.get_shape())
        conv1 = tf.concat([conv1_a, conv1_b, conv1_c], 4)
        with tf.variable_scope(self.param['mode']+'conv2') as scope:
            conv2 = self.cnnutils.conv_relu(input_=conv1,
                                       kernel_shape=[5, 5, 5, 45, 60],
                                       biases_shape=[60], scope=scope)
        print("Conv2", conv2.get_shape())
        with tf.variable_scope(self.param['mode']+'conv3') as scope:
            conv3 = self.cnnutils.conv_relu(input_=conv2,
                                   kernel_shape=[5, 5, 5, 60, 64],
                                   biases_shape=[64], scope=scope)
        print("Conv3", conv3.get_shape())
        with tf.variable_scope(self.param['mode']+'conv4') as scope:
            conv4 = self.cnnutils.conv_relu(input_=conv3,
                                   kernel_shape=[3, 3, 3, 64, 100],
                                   biases_shape=[100], scope=scope)
        print("Conv4", conv4.get_shape())

        with tf.variable_scope(self.param['mode']+'conv5') as scope:
            conv5 = self.cnnutils.conv_relu(input_=conv4,
                                   kernel_shape=[3, 3, 3, 100, 128],
                                   biases_shape=[128], scope=scope)
        print("Conv5", conv5.get_shape())
        with tf.variable_scope(self.param['mode']+'conv6') as scope:
            conv6 = self.cnnutils.conv_relu(input_=conv5,
                                   kernel_shape=[3, 3, 3, 128, 256],
                                   biases_shape=[256], scope=scope)
        print("Conv6", conv6.get_shape())
        with tf.variable_scope(self.param['mode']+'conv7') as scope:
            conv7 = self.cnnutils.conv_relu(input_=conv6,
                                   kernel_shape=[3, 3, 3, 256, 512],
                                   biases_shape=[512], scope=scope)
        print("Conv7", conv7.get_shape())

        with tf.variable_scope(self.param['mode']+'fullcn') as scope:
            vector_per_batch = tf.reshape(conv7, [self.param['batch_size'],
                                          -1])
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                 shape=[512, 512])
            biases = self.cnnutils.variable_on_gpu(name="biases",
                                          shape=[512],
                                          initializer=tf.constant_initializer(
                                              0.01))
            pre_activation = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope(self.param['mode']+'logits') as scope:
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                 shape=[512, self.param[
                                                     'classes']])
            biases = self.cnnutils.variable_on_gpu(name='biases',
                                          shape=[self.param['classes']],
                                          initializer=tf.constant_initializer(
                                              0.01))
            logits = tf.add(tf.matmul(fullcn_drop, weights), biases,
                            name=scope.name)
            print('logits:', logits.get_shape())

        return logits

    def get_features(self, sess, saver):
        with tf.Graph().as_default() as model_graph:
            ckpath = self.param['transfer_checkpoint_path']
            ckpt = tf.train.get_checkpoint_state(ckpath)
            print(ckpath, ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, train_data, validation_data, test):
        """
        This function starts building and
        training the 3D CNN model.

        Args:
            train_data: the training dataset object of class DataInput
            validation_data: validation dataset object of class DataInput
            test: Data object of DataInput class to test the model accuracy
        """
        mode = self.param['mode'] 
        with tf.Graph().as_default():

            images = tf.placeholder(dtype=tf.float32,
                                    shape=[None,
                                           self.param['depth'],
                                           self.param['height'],
                                           self.param['width'],
                                           self.param['channels']],
                                    name=mode+'images')
            if self.mlp == 'True':
                images = tf.placeholder(dtype=tf.float32,
                                        shape=[None,
                                               self.param['num_features']],
                                        name=mode + 'images')
            labels = tf.placeholder(dtype=tf.int8,
                                    shape=[None, self.param['classes']],
                                    name=mode+'labels')
            keep_prob = tf.placeholder(tf.float32, name=mode+'keep_prob')
            global_step = tf.get_variable(name=mode+'global_step',
                                          shape=[],
                                          initializer=tf.constant_initializer(
                                              0),
                                          trainable=False)
            num_batches_epoch = self.cnnutils.num_steps(dataset=train_data)
            num_steps = num_batches_epoch* self.param['num_epochs']
            print("Total Numsteps: ", num_steps, flush=True)

            learn_rate = self.param['learning_rate']

            if self.param['decay_lr'] == 'True': # Decay every epoch
                learn_rate = tf.train.exponential_decay(
                    learning_rate=self.param['learning_rate'],
                    global_step=global_step,
                    decay_steps=num_batches_epoch,
                    decay_rate=self.param['decay_factor'],
                    staircase=True)

            opt = tf.train.AdamOptimizer(
                learning_rate=learn_rate)

            # tf.summary.scalar('learning_rate', learn_rate)
            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train' + mode) as scope:
                    logits = []
                    if self.mlp == 'False':
                        logits = self.inference(images, keep_prob)
                    elif self.mlp == 'True':
                        logits = self.mlp_inference(images, keep_prob)
                    self.cnnutils.inference_loss(logits, labels)

                    losses = tf.get_collection('losses', scope)

                    # Sum all the losses
                    total_loss = tf.add_n(losses, name='total_loss')
                    xloss = tf.get_collection('crossloss', scope)[0]
                    l2loss = tf.add_n(tf.get_collection('l2loss', scope))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = opt.minimize(total_loss,
                                            global_step=global_step)

                # Evaluation
                correct_prediction = tf.equal(tf.argmax(labels, 1),
                                              tf.argmax(logits, 1))
                eval_op = tf.reduce_sum(tf.cast(correct_prediction,
                                                tf.float32))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))
                tf.summary.scalar('train_batch_accuracy', accuracy)

                #Setting the random seed
                tf.set_random_seed(0)

                saver = tf.train.Saver(max_to_keep=None)
                init = tf.global_variables_initializer()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                sess = tf.InteractiveSession(config=config)
                sess.run(init)
                if self.param['transfer'] == 'True':
                    tr_depth = 2 * (9 - self.param['transfer_depth'])
                    train_objects = tf.trainable_variables()[:-tr_depth]
                    if tr_depth == 0:
                        train_objects = tf.trainable_variables()
                    dict_map = {v.name[:-2]: v for v in train_objects}
                    print("Trainable variables:", dict_map, flush=True)
                    print("Not used:", [v.name for v in
                                        tf.trainable_variables()[-tr_depth:]])
                    saver = tf.train.Saver(dict_map, max_to_keep=None)
                    self.get_features(sess, saver)
                summary_writer = tf.summary.FileWriter(
                    self.param['summary_path'], sess.graph)
                summary_op = tf.summary.merge_all()
                tf.get_default_graph().finalize()

                for i in range(0, len(validation_data.files)):
                    validation_data.batch_index[i] = 0
                    validation_data.shuffle()
                    train_data.batch_index[i] = 0
                    train_data.shuffle()

                for step in range(1, num_steps):
                    print("Step:", step, "Total:", num_steps)
                    start_time = time.time()

                    _, image_data, label_data = train_data.next_batch()
                    summary_values, _, loss_value, cross_loss, l2_loss = \
                        sess.run(
                        [summary_op, train_op, total_loss, xloss, l2loss],
                        feed_dict={
                            images: image_data,
                            labels: label_data,
                            keep_prob: self.param['keep_prob']
                        }
                    )
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps / 2:
                        checkpoint_path = self.param['checkpoint_path'] + \
                                          'model.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                        print("Saving checkpoint model...")
                        sys.stdout.flush()
                    if step % 50 == 0:
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

                        num_examples_per_step = self.param['batch_size']
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration
                        format_str = (
                        '%s: step %d, loss = %.2f %.2f %.2f (%.1f '
                        'examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                                            cross_loss, l2_loss,
                                            examples_per_sec, sec_per_batch))
                        sys.stdout.flush()
                    # summary_writer.add_summary(summary_values, step)

                    # Saving Model Checkpoints for evaluation
                    if step % num_batches_epoch == 0 or (
                        step + 1) == num_steps:
                        for i in range(0, len(validation_data.files)):
                            validation_data.batch_index[i] = 0
                            validation_data.shuffle()
                            train_data.batch_index[i] = 0
                            train_data.shuffle()

                        # Evaluate against the training data.
                        print("Step: %d Training accuracy: %g " %
                              (step, self.cnnutils.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=train_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss,
                                                     xloss=xloss,
                                                     l2loss=l2loss)))

                        # Evaluate against the validation data
                        print("Step: %d Validation accuracy: %g" %
                              (step, self.cnnutils.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=images,
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss,
                                                     xloss=xloss,
                                                     l2loss=l2loss)))
                        for i in range(0, len(validation_data.files)):
                            validation_data.batch_index[i] = 0
                            validation_data.shuffle()
                            train_data.batch_index[i] = 0
                            train_data.shuffle()
                    sys.stdout.flush()
                if test == True:
                    ckpt = tf.train.get_checkpoint_state(
                            self.param['checkpoint_path'])
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print("Testing model")
                        print(self.cnnutils.evaluation(sess=sess,
                                                 eval_op=eval_op,
                                                 dataset=validation_data,
                                                 images=images,
                                                 labels=labels,
                                                 keep_prob=keep_prob,
                                                 corr=correct_prediction,
                                                 loss=total_loss,
                                                  xloss=xloss,
                                                  l2loss=l2loss))
                    else:
                        print("No checkpoint found.")

