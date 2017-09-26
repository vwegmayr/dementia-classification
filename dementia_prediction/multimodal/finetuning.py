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
from dementia_prediction.cnn_utils import CNNUtils


class CNNMultimodal:
    """
    This class fuses the baseline architectures at the 7th conv layer. By 
    default, the architecture is trained from scratch. If finetuning is enabled,
    the weights are finetuned.
    """

    def __init__(self, params):
        self.param = params['cnn']
        self.modalities = params['cnn']['mode']
        self.cnnutils = CNNUtils(params)
        self.checkpoints = {0: params['cnn']['ckpt_mode1'],
                           1: params['cnn']['ckpt_mode2'],
                           2: params['cnn']['ckpt_mode3']
                           }
        self.models = {0: params['cnn']['meta_mode1'],
                       1: params['cnn']['meta_mode2'],
                       2: params['cnn']['meta_mode3']
                      }
        print("Multimodal", params, flush=True)


    def get_conv7(self, images, keep_prob, mode):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print(images.get_shape())
        prefix = self.modalities[mode]
        with tf.variable_scope(prefix+'conv1_a') as scope:
            conv1_a = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[5, 5, 5, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_a", conv1_a.get_shape())
        with tf.variable_scope(prefix+'conv1_b') as scope:
            conv1_b = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[6, 6, 6, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_b", conv1_b.get_shape())
        with tf.variable_scope(prefix+'conv1_c') as scope:
            conv1_c = self.cnnutils.conv_relu(input_=images,
                                         kernel_shape=[7, 7, 7, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_c", conv1_c.get_shape())
        conv1 = tf.concat([conv1_a, conv1_b, conv1_c], 4)
        with tf.variable_scope(prefix+'conv2') as scope:
            conv2 = self.cnnutils.conv_relu(input_=conv1,
                                       kernel_shape=[5, 5, 5, 45, 60],
                                       biases_shape=[60], scope=scope)
        print("Conv2", conv2.get_shape())
        with tf.variable_scope(prefix+'conv3') as scope:
            conv3 = self.cnnutils.conv_relu(input_=conv2,
                                   kernel_shape=[5, 5, 5, 60, 64],
                                   biases_shape=[64], scope=scope)
        print("Conv3", conv3.get_shape())
        with tf.variable_scope(prefix+'conv4') as scope:
            conv4 = self.cnnutils.conv_relu(input_=conv3,
                                   kernel_shape=[3, 3, 3, 64, 100],
                                   biases_shape=[100], scope=scope)
        print("Conv4", conv4.get_shape())
        with tf.variable_scope(prefix+'conv5') as scope:
            conv5 = self.cnnutils.conv_relu(input_=conv4,
                                   kernel_shape=[3, 3, 3, 100, 128],
                                   biases_shape=[128], scope=scope)
        print("Conv5", conv5.get_shape())
        with tf.variable_scope(prefix+'conv6') as scope:
            conv6 = self.cnnutils.conv_relu(input_=conv5,
                                   kernel_shape=[3, 3, 3, 128, 256],
                                   biases_shape=[256], scope=scope)
        print("Conv6", conv6.get_shape())
        with tf.variable_scope(prefix+'conv7') as scope:
            conv7 = self.cnnutils.conv_relu(input_=conv6,
                                   kernel_shape=[3, 3, 3, 256, 512],
                                   biases_shape=[512], scope=scope)
        print("Conv7", conv7.get_shape())
        return conv7

    def get_conv1(self, images, keep_prob, mode):
        """
        This function builds the 3D Convolutional Neural Network architecture
        Args:
            images: Input MR Images

        Returns:
            Logits calculated at the last layer of the 3D CNN.
        """
        print(images.get_shape())
        prefix = self.modalities[mode]
        # Change 7,7,7 to 5,5,5
        with tf.variable_scope(prefix+'conv1_a') as scope:
            conv1_a = self.cnnutils.conv_relu(input_=images,
                                     kernel_shape=[5, 5, 5, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_a", conv1_a.get_shape())
        with tf.variable_scope(prefix+'conv1_b') as scope:
            conv1_b = self.cnnutils.conv_relu(input_=images,
                                     kernel_shape=[6, 6, 6, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_b", conv1_b.get_shape())
        with tf.variable_scope(prefix+'conv1_c') as scope:
            conv1_c = self.cnnutils.conv_relu(input_=images,
                                     kernel_shape=[7, 7, 7, self.param[
                                             'channels'], 15],
                                         biases_shape=[15], scope=scope)
        print("Conv1_c", conv1_c.get_shape())
        conv1 = tf.concat([conv1_a, conv1_b, conv1_c], 4)
        return conv1

    def fusion_conv1(self, fusion_input, keep_prob):
        prefix = 'Fusion'
        with tf.variable_scope(prefix + 'conv2') as scope:
            conv2 = self.cnnutils.conv_relu(input_=fusion_input,
                                            kernel_shape=[5, 5, 5, 135, 135],
                                            biases_shape=[135], scope=scope)
        print("Conv2", conv2.get_shape())
        with tf.variable_scope(prefix + 'conv3') as scope:
            conv3 = self.cnnutils.conv_relu(input_=conv2,
                                            kernel_shape=[5, 5, 5, 135, 140],
                                            biases_shape=[140], scope=scope)
        print("Conv3", conv3.get_shape())
        with tf.variable_scope(prefix + 'conv4') as scope:
            conv4 = self.cnnutils.conv_relu(input_=conv3,
                                            kernel_shape=[3, 3, 3, 140, 145],
                                            biases_shape=[145], scope=scope)
        print("Conv4", conv4.get_shape())
        with tf.variable_scope(prefix + 'conv5') as scope:
            conv5 = self.cnnutils.conv_relu(input_=conv4,
                                            kernel_shape=[3, 3, 3, 145, 150],
                                            biases_shape=[150], scope=scope)
        print("Conv5", conv5.get_shape())
        with tf.variable_scope(prefix + 'conv6') as scope:
            conv6 = self.cnnutils.conv_relu(input_=conv5,
                                            kernel_shape=[3, 3, 3, 150, 256],
                                            biases_shape=[256], scope=scope)
        print("Conv6", conv6.get_shape())
        with tf.variable_scope(prefix + 'conv7') as scope:
            conv7 = self.cnnutils.conv_relu(input_=conv6,
                                            kernel_shape=[3, 3, 3, 256, 512],
                                            biases_shape=[512], scope=scope)
        print("Conv7", conv7.get_shape())
        with tf.variable_scope(prefix + 'fullcn') as scope:
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

        with tf.variable_scope(prefix + 'logits') as scope:
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                          shape=[512,
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

    def fusion_conv7(self, fusion_input, keep_prob):
        with tf.variable_scope('Fusionfullcn') as scope:
            vector_per_batch = tf.reshape(fusion_input, [self.param['batch_size'],
                                                  -1])
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                 shape=[1536, 512])
            biases = self.cnnutils.variable_on_gpu(name="biases",
                                          shape=[512],
                                          initializer=tf.constant_initializer(
                                              0.01))
            pre_activation = tf.matmul(vector_per_batch, weights) + biases
            fullcn = tf.nn.relu(pre_activation, name=scope.name)
            fullcn_drop = tf.nn.dropout(fullcn, keep_prob)
            print('fullcn:', fullcn_drop.get_shape())

        with tf.variable_scope('Fusionlogits') as scope:
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



    def get_features(self, sess, saver, mode):
        with tf.Graph().as_default() as model_graph:
            ckpath = self.checkpoints[mode]
            ckpt = self.models[mode]
            print(ckpath, ckpt)
            if ckpt:
                saver.restore(sess, ckpt)
                print("Restoring", ckpt)

    def evaluation(self, sess, eval_op, dataset, images, labels,
                   keep_prob, loss, xloss, l2loss, corr):
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
        images1 = images[0]
        images2 = images[1]
        images3 = images[2]
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
        for step in range(num_steps):
            patients, image_data, label_data = dataset.next_batch()
            predictions, correct_, loss_, xloss_, l2loss_ = \
                sess.run([eval_op, corr, loss, xloss, l2loss],
                feed_dict={
                    images1: image_data[0],
                    images2: image_data[1],
                    images3: image_data[2],
                    labels: label_data,
                    keep_prob: 1.0
                })
            print("Prediction:", correct_)
            correct_predictions += predictions
            pred_out.update(dict(zip(patients, correct_)))
            total_seen += self.param['batch_size']
            print("Accuracy until "+str(total_seen)+" data points is: " +
                      str(correct_predictions/total_seen))
            print("loss", loss_, xloss_, l2loss_)
        accuracy_ = 0
        for key, value in pred_out.items():
            if value == True:
                accuracy_ += 1
        accuracy_ /= len(pred_out)
        print("Accuracy of ", len(pred_out)," images is ", accuracy_)
        # TODO: Add accuracy [2]
        sys.stdout.flush()
        return accuracy_

    def train(self, train_data, validation_data, test):
        """
        This function creates the training operations and starts building and
        training the 3D CNN model.

        Args:
            train_data: the training data required for 3D CNN
            validation_data: validation data to test the accuracy of the model.

        """
        mode = 'Fusion'
        with tf.Graph().as_default():
            images1 = tf.placeholder(dtype=tf.float32,
                                    shape=[None,
                                           self.param['depth'],
                                           self.param['height'],
                                           self.param['width'],
                                           self.param['channels']],
                                     name=self.modalities[0]+'images')
            images2 = tf.placeholder(dtype=tf.float32,
                                     shape=[None,
                                            self.param['depth'],
                                            self.param['height'],
                                            self.param['width'],
                                            self.param['channels']],
                                     name=self.modalities[1]+'images')

            images3 = tf.placeholder(dtype=tf.float32,
                                     shape=[None,
                                            self.param['depth'],
                                            self.param['height'],
                                            self.param['width'],
                                            self.param['channels']],
                                     name=self.modalities[2] + 'images')

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
            num_steps = num_batches_epoch * self.param['num_epochs']
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

            tf.summary.scalar('learning_rate', learn_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train'+mode) as scope:
                    logits = []
                    if self.param['fusion_layer'] == 'conv7':
                        conv_1 = self.get_conv7(images1, keep_prob, 0)
                        conv_2 = self.get_conv7(images2, keep_prob, 1)
                        conv_3 = self.get_conv7(images3, keep_prob, 2)
                        conv_fusion = tf.concat([conv_1, conv_2,
                                                  conv_3], 4)
                        logits = self.fusion_conv7(conv_fusion, keep_prob)
                    elif self.param['fusion_layer'] == 'conv1':
                        conv_1 = self.get_conv1(images1, keep_prob, 0)
                        conv_2 = self.get_conv1(images2, keep_prob, 1)
                        conv_3 = self.get_conv1(images3, keep_prob, 2)
                        conv_fusion = tf.concat([conv_1, conv_2,
                                                  conv_3], 4)
                        logits = self.fusion_conv1(conv_fusion, keep_prob)

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
                if self.param['fusion'] == 'finetune':
                    train_objects = tf.trainable_variables()[:-2]
                    dict_map_T1 =  {v.name[:-2]:v for v in train_objects
                                    if self.param['mode1'] in v.name[:-2]}
                    dict_map_T2 =  {v.name[:-2]:v for v in train_objects
                                    if self.param['mode2'] in v.name[:-2]}
                    dict_map_DTI_FA =  {v.name[:-2]:v for v in train_objects
                                        if self.param['mode3'] in v.name[:-2]}
                    print("Trainable variables:",
                          dict_map_T1, dict_map_T2, dict_map_DTI_FA)
                    saver1 = tf.train.Saver(dict_map_T1)
                    saver2 = tf.train.Saver(dict_map_T2)
                    saver3 = tf.train.Saver(dict_map_DTI_FA)
                    self.get_features(sess, saver1, 0)
                    self.get_features(sess, saver2, 1)
                    self.get_features(sess, saver3, 2)

                # Create a summary writer
                summary_writer = tf.summary.FileWriter(
                    self.param['summary_path'], sess.graph)
                summary_op = tf.summary.merge_all()
                tf.get_default_graph().finalize()

                for i in range(0, len(validation_data.files)):
                    validation_data.batch_index[i] = 0
                    train_data.batch_index[i] = 0
                    validation_data.shuffle()
                    train_data.shuffle()
                for step in range(1, num_steps):
                    print("Step:", step, "/", num_steps)
                    
                    start_time = time.time()
                    _, image_data, label_data = train_data.next_batch()
                    summary_values, _, loss_value, cross_loss, l2_loss = \
                        sess.run(
                        [summary_op,
                         train_op,
                         total_loss,
                         xloss,
                         l2loss],
                        feed_dict={
                            images1: image_data[0],
                            images2: image_data[1],
                            images3: image_data[2],
                            labels: label_data,
                            keep_prob: self.param['keep_prob']
                        }
                    )
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps:
                        checkpoint_path = self.param['checkpoint_path'] + \
                                          'multimodal_model.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                    if step % 50 == 0:
                        accuracy_ = sess.run(accuracy,
                                             feed_dict={
                                                 images1: image_data[0],
                                                 images2: image_data[1],
                                                 images3: image_data[2],
                                                 labels: label_data,
                                                 keep_prob: 1.0
                                             })
                        print("Train Batch Accuracy. %g step %d" % (accuracy_,
                                                                    step))
                        duration = time.time() - start_time
                        time_left = (num_steps - step) * duration / 60
                        print("Time left (min):", time_left)
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
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=train_data,
                                                     images=[images1,
                                                             images2,images3],
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss,
                                                     xloss=xloss,
                                                     l2loss=l2loss)))
                        # Evaluate against the validation data
                        print("Step: %d Validation accuracy: %g" %
                              (step, self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=[images1,
                                                             images2,images3],
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
                        print(self.evaluation(sess=sess,
                                                     eval_op=eval_op,
                                                     dataset=validation_data,
                                                     images=[images1,
                                                             images2,images3],
                                                     labels=labels,
                                                     keep_prob=keep_prob,
                                                     corr=correct_prediction,
                                                     loss=total_loss,
                                                     xloss=xloss,
                                                     l2loss=l2loss))
                    else:
                        print("No checkpoint found.")

