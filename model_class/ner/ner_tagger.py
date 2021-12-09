#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics
import utils.utils as utils
import codecs
from utils.logger import Logger
import model_class.neural_tagger as neural_tagger


class NERTagger(neural_tagger.NeuralTagger):
    def __init__(self, mbedding_matrix, args):
        super(NERTagger, self).__init__(mbedding_matrix, args)


    def common_build(self, args):
        self.dic_dim = 0 if args == None or args.use_dics_flag == False else args.num_dic_feature_classes

        with tf.name_scope("inputs"): # self.time_step_size
            self.X = tf.placeholder(tf.int32, shape = [None, None, self.feature_window_size], name = "X_placeholder")
            self.DIC = tf.placeholder(tf.float32, shape = [None, None, self.dic_dim], name = "D_placeholder")
            self.Y = tf.placeholder(tf.int32, shape = [None, None], name = "Y_placeholder")
            self.X_len = tf.placeholder(tf.int32, shape = [None], name = "X_len_placeholder")
            self.keep_prob = tf.placeholder(tf.float32, name = "output_dropout")


    def inference_with_dic(self, X, DIC, X_len):
        pass


    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)

            if self.fine_tune_flag:
                regularization += tf.nn.l2_loss(self.embedding_matrix)

            cost += regularization * self.l2_reg
            return cost


    def run(self, train_set, valid_set, test_set, args = None):
        if args is None:
            Logger.info("args ERROR")
            sys.exit(0)

        train_x = train_set.feature_list
        train_dic = train_set.dic_feature_one_hot
        train_y = train_set.word_label_list
        train_lens = train_set.sentence_lengths

        valid_x = valid_set.feature_list
        valid_dic = valid_set.dic_feature_one_hot
        valid_y = valid_set.word_label_list
        valid_lens = valid_set.sentence_lengths

        test_x = test_set.feature_list
        test_dic = test_set.dic_feature_one_hot
        test_y = test_set.word_label_list
        test_lens = test_set.sentence_lengths

        self.lr = args.lr
        self.training_iter = args.train_steps
        self.train_file_path = args.train_data
        self.test_file_path = args.valid_data
        self.display_step = args.display_step

        # unary_scores & loss
        pred = None
        if args.use_dics_flag == True:
            pred = self.inference_with_dic(self.X, self.DIC, self.X_len)
        else:
            pred = self.inference(self.X, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="train_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('summary'):
            if args.log_tensorboard:
                # localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = args.log_dir_

                info = 'batch{}, lr{}, l2_reg{}'.format(self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + self.test_file_path + ';' + 'Method:%s' % self.__str__()
                train_acc = tf.placeholder(tf.float32)
                train_loss = tf.placeholder(tf.float32)
                summary_acc = tf.summary.scalar('ACC ' + info, train_acc)
                summary_loss = tf.summary.scalar('LOSS ' + info, train_loss)
                summary_op = tf.summary.merge([summary_loss, summary_acc])

                valid_acc = tf.placeholder(tf.float32)
                valid_loss = tf.placeholder(tf.float32)
                summary_valid_acc = tf.summary.scalar('ACC ' + info, valid_acc)
                summary_valid_loss = tf.summary.scalar('LOSS ' + info, valid_loss)
                summary_valid = tf.summary.merge([summary_valid_loss, summary_valid_acc])

        with tf.name_scope('saveModel'):
            # localtime = time.strftime("%X-%Y-%m-%d", time.localtime())
            saver = tf.train.Saver(max_to_keep = 5)
            save_dir = args.model_dir
            utils.makedirs(save_dir)

        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(Summary_dir + '/train', sess.graph)
            valid_summary_writer = tf.summary.FileWriter(Summary_dir + '/valid', sess.graph)

            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, args.restore_model)
                Logger.info("[+] Model restored from %s" % args.restore_model)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in xrange(self.training_iter):
                for train, num in self.get_batch_data(train_x, train_dic, train_y, train_lens, self.batch_size, (1 - self.dropout), use_dics_flag = args.use_dics_flag):
                    _, step, trans_matrix, loss, predication = sess.run(
                        [optimizer, global_step, self.transition, cost, pred], feed_dict=train)
                    tags_seqs, _ = utils.viterbi_decode(num, predication, train[self.X_len], trans_matrix)
                    f = self.evaluate(num, tags_seqs, train[self.Y], train[self.X_len])

                    if args.log_tensorboard:
                        summary = sess.run(summary_op, feed_dict={train_loss: loss, train_acc: f})
                        train_summary_writer.add_summary(summary, step)

                    Logger.info('Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, f))

                save_path = saver.save(sess, save_dir + "model.ckpt", global_step=step)
                Logger.info("[+] Model saved in file: %s" % save_path)

                if epoch % self.display_step == 0:
                    rd, loss, acc = 0, 0., 0.
                    for valid, num in self.get_batch_data(valid_x, valid_dic, valid_y, valid_lens, self.batch_size, use_dics_flag = args.use_dics_flag):
                        trans_matrix, _loss, predication = sess.run(
                            [self.transition, cost, pred], feed_dict=valid)
                        loss += _loss
                        tags_seqs, _ = utils.viterbi_decode(num, predication, valid[self.X_len], trans_matrix)
                        f = self.evaluate(num, tags_seqs, valid[self.Y], valid[self.X_len])
                        acc += f
                        rd += 1

                    loss /= rd
                    acc /= rd
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                        
                    if args.log_tensorboard:
                        summary = sess.run(summary_valid, feed_dict={
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)

                    Logger.info('----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
                    Logger.info('Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}'.format(step, loss, acc))
                    Logger.info('round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter))
            Logger.info('Optimization Finished!')

            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_dic, test_y, test_lens, self.batch_size, shuffle=False, use_dics_flag = args.use_dics_flag):
                trans_matrix, _loss, predication = sess.run(
                    [self.transition, cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                tags_seqs, _ = utils.viterbi_decode(num, predication, test[self.X_len], trans_matrix)
                f = self.evaluate(num, tags_seqs, test[self.Y], test[self.X_len])
                acc += f
                pred_test_y.extend(tags_seqs)
            acc /= rd
            loss /= rd
            return pred_test_y, loss, acc


    def get_batch_data(self, x, dic, y, l, batch_size, keep_prob = 1.0, shuffle = True, use_dics_flag = False):
        for index in utils.batch_index(len(y), batch_size, 1, shuffle):
            if use_dics_flag == False:
                feed_dict = {
                    self.X: x[index],
                    self.Y: y[index],
                    self.X_len: l[index],
                    self.keep_prob: keep_prob,
                }
            else:
                feed_dict = {
                    self.X: x[index],
                    self.DIC: dic[index],
                    self.Y: y[index],
                    self.X_len: l[index],
                    self.keep_prob: keep_prob,
                }

            yield feed_dict, len(index)


    def evaluate(self, num, labels, y, y_lens):
        golds = []
        preds = []
        for i in xrange(num):
            p_len = y_lens[i]
            golds.extend(y[i][:p_len])
            preds.extend(labels[i])
        # p = metrics.precision_score(golds, preds, average='macro')
        # r = metrics.recall_score(golds, preds, average='macro')
        # f = metrics.f1_score(golds, preds, average='macro')
        # return (p, r, f)
        return metrics.precision_score(golds, preds, average='micro')
