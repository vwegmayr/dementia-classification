from datetime import datetime
import time
import math
import tensorflow as tf
import numpy as np
import sys
from dementia_prediction.cnn_utils import CNNUtils

class TransferToptune:
    """
    To train the network and evaluate it, initialize the class with the
    required parameter file and call the function train() with training and
    validation datasets.
    """

    def __init__(self, params):
        self.param = params['cnn']
        self.cnnutils = CNNUtils(params)
        print("Toptuning", params, flush=True)

    def inference_fullcn(self, transfer_input):
        with tf.variable_scope('logits') as scope:
            weights = self.cnnutils.weight_decay_variable(name="weights",
                                                 shape=[512, 2])
            biases = self.cnnutils.variable_on_gpu(name='biases',
                                          shape=[2],
                                          initializer=tf.constant_initializer(
                                              0.01))
            logits = tf.add(tf.matmul(transfer_input, weights), biases,
                            name=scope.name)
        return logits

    def evaluation(self, sess, eval_op, dataset, images, transfer_input,
                   labels, keep_prob, loss, xloss, l2loss, corr):
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
            layer_path = ''
            if self.param['transfer'] == 'fullcn':
                layer_path = 'TrainADNI_T1/ADNI_T1fullcn2/ADNI_T1fullcn2:0'
            elif self.param['transfer'] == 'conv1':
                layer_path = 'conv1'
            features_images = self.cnnutils.get_features("ADNI_T1",
                                                         self.param[
                                                             'meta_path'],
                                                         image_data,
                                                         layer_path)
            predictions, correct_, loss_, xloss_, l2loss_ = \
                sess.run([eval_op, corr, loss, xloss, l2loss],
                feed_dict={
                    transfer_input: features_images,
                    labels: label_data,
                    keep_prob: 1.0
                })
            print("Prediction:", correct_)
            correct_predictions += predictions
            pred_out.update(dict(zip(patients, correct_)))
            total_seen += self.param['batch_size']
            print("Accuracy until "+str(total_seen)+" data points is: " +
                      str(correct_predictions/total_seen))
            print("loss", loss_)
            #print("logits:", logits_)
        accuracy_ = 0
        for key, value in pred_out.items():
            if value == True:
                accuracy_ += 1
        accuracy_ /= len(pred_out)
        print("Accuracy of ", len(pred_out)," images is ", accuracy_)
        sys.stdout.flush()
        return accuracy_

    def train(self, train_data, validation_data, test):
        """
        This function creates the training operations and starts building and
        training the 3D CNN model.

        Args:
            train_data: the training data required for 3D CNN
            validation_data: validation data to test the accuracy of the model.
            test: Test the final model on the validation data
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
            if self.param['decay_lr'] == 'True':
                learn_rate = tf.train.exponential_decay(
                    learning_rate=self.param['learning_rate'],
                    global_step=global_step,
                    decay_steps=num_batches_epoch,
                    decay_rate=self.param['decay_factor'],
                    staircase=True)

            opt = tf.train.AdamOptimizer(
                learning_rate=learn_rate)

            with tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('Train'+mode) as scope:
                    logits = []
                    if self.param['transfer'] == 'fullcn':
                        transfer_input = tf.placeholder(dtype=tf.float32,
                                                shape = [None, 512],
                                                name=mode + 'transfer_input')
                        logits = self.inference_fullcn(transfer_input)

                        self.cnnutils.inference_loss(logits, labels)
                    if self.param['transfer'] == 'conv1':
                        transfer_input = tf.placeholder(dtype=tf.float32,
                                                        shape=[None, 46, 55,
                                                               46, 45],
                                                name=mode + 'transfer_input')
                        logits = self.cnnutils.inference_conv2(
                            transfer_input, keep_prob)
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

                # Create a summary writer
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
                    print("Step:", step,"Total:", num_steps)
                    start_time = time.time()
                    _, image_data, label_data = train_data.next_batch()
                    layer_path = ''
                    if self.param['transfer'] == 'fullcn':
                        layer_path = 'TrainADNI_T1/ADNI_T1fullcn2/ADNI_T1fullcn2:0'
                    elif self.param['transfer'] == 'conv1':
                        layer_path = 'conv1'
                    features_images = self.cnnutils.get_features("ADNI_T1",
                                                                 self.param[
                                                                     'meta_path'],
                                                                 image_data,
                                                                 layer_path)
                    summary_values, _, loss_value, cross_loss, l2_loss = sess.run(
                        [summary_op,
                         train_op,
                         total_loss,
                         xloss,
                         l2loss],
                        feed_dict={
                            transfer_input: features_images,
                            labels: label_data,
                            keep_prob: self.param['keep_prob']
                        }
                    )
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps:
                        checkpoint_path = self.param['checkpoint_path'] + \
                                          'transfer_model.ckpt'
                        saver.save(sess, checkpoint_path, global_step=step)
                    duration = time.time() - start_time

                    assert not np.isnan(
                        loss_value), 'Model diverged with loss = NaN'

                    if step % 50 == 0:
                        accuracy_ = sess.run(accuracy,
                                         feed_dict={
                                             transfer_input: features_images,
                                             labels: label_data,
                                             keep_prob: 1.0
                                         })
                        print("Train Batch Accuracy. %g step %d" % (accuracy_,
                                                                    step))
                        num_examples_per_step = self.param['batch_size']
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration

                        format_str = ('%s: step %d, loss = %.2f %.2f %.2f (%.1f '
                                      'examples/sec; %.3f sec/batch)')
                        print(format_str % (datetime.now(), step, loss_value,
                               cross_loss, l2_loss, examples_per_sec, sec_per_batch))
                        sys.stdout.flush()
                        #summary_writer.add_summary(summary_values, step)

                    # Saving Model Checkpoints for evaluation
                    if step % num_batches_epoch == 0 or (step + 1) == num_steps:
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
                                                     images=images,
                                                     transfer_input=transfer_input,
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
                                                     images=images,
                                                     transfer_input=transfer_input,
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
                '''
                if test == True:
                    ckpt = tf.train.get_checkpoint_state(
                        self.param['checkpoint_path'])
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
