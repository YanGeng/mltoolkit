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

class ClassifyTagger(neural_tagger.NeuralTagger):
    def __init__(self, mbedding_matrix, args):
        super(ClassifyTagger, self).__init__(mbedding_matrix, args)


    def common_build(self, args):
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.int32, shape = [None, self.time_step_size, self.feature_window_size], name = "X_placeholder")
            self.Y = tf.placeholder(tf.int32, shape = [None, self.num_classes], name = "Y_placeholder")
            # self.X_len = tf.placeholder(tf.int32, shape = [None, ], name = "X_len_placeholder")
            self.keep_prob = tf.placeholder(tf.float32, name = "output_dropout")


    # def build(self):
    #     pass


    # def inference(self, X, X_len = None, reuse = None):
    #     pass


    def loss(self, pred):
        with tf.name_scope('loss'):
            cost = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = self.Y)
            cost = tf.reduce_mean(cost)
            regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)

            if self.fine_tune_flag:
                regularization += tf.nn.l2_loss(self.embedding_matrix)

            cost += regularization * self.l2_reg
            return cost


    # def predict(self, X, X_len, model_dir):
    #     # inference_res = self.inference(X, X_len)

    #     # /Users/tony/Sandbox-Self/MLT/models/11:42:15-2020-06-12/model.ckpt-50
    #     saver = tf.train.import_meta_graph(model_dir + '.meta')
    #     init = tf.global_variables_initializer()
    #     # saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         saver.restore(sess, model_dir)
    #         sess.run(init)
    #         graph = tf.get_default_graph()

    #         inference_res = graph.get_tensor_by_name("line_transform/inference_result:0")
    #         trans1 = graph.get_tensor_by_name("transitions:0")
    #         trans2 = graph.get_tensor_by_name("loss/my_transition:0")
    #         X_placeholder = graph.get_tensor_by_name("inputs/X_placeholder:0")
    #         X_len_placeholder = graph.get_tensor_by_name("inputs/X_len_placeholder:0")
    #         output_dropout = graph.get_tensor_by_name("inputs/output_dropout:0")
    #         W = graph.get_tensor_by_name("weights_1:0")
    #         B = graph.get_tensor_by_name("biases/bias:0")

    #         dicts = {
    #             X_placeholder: X[[0]],
    #             # self.Y: [],
    #             X_len_placeholder: X_len[[0]],
    #             output_dropout: 1.0,
    #         }

    #         tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    #         with codecs.open("/Users/tony/test/tensor_name", 'w', 'utf-8') as out:
    #             for tensor_name in tensor_name_list:
    #                 print(tensor_name,'\n')
    #                 out.write("%s\n" % tensor_name.encode("utf-8"))
            
    #         trans_matrix1, trans_matrix2, predication = sess.run([trans1, trans2, inference_res], dicts)
    #         tags_seqs, _ = utils.viterbi_decode(len(X), predication, X_len, trans_matrix1)

    #         pred_y = []
    #         pred_y.extend(tags_seqs)

    #         return pred_y


    def run(self, train_set, valid_set, test_set, args = None):
        if args is None:
            Logger.info("args ERROR")
            sys.exit(0)

        train_x = train_set.feature_list
        train_y = train_set.sentence_label_one_hot
        train_lens = train_set.sentence_lengths

        valid_x = valid_set.feature_list
        valid_y = valid_set.sentence_label_one_hot
        valid_lens = valid_set.sentence_lengths

        test_x = test_set.feature_list
        test_y = test_set.sentence_label_one_hot
        test_lens = test_set.sentence_lengths

        self.lr = args.lr
        self.training_iter = args.train_steps
        self.train_file_path = args.train_data
        self.test_file_path = args.valid_data
        self.display_step = args.display_step

        # unary_scores & loss
        pred, _ = self.inference(self.X, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="train_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost, global_step = global_step)

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
                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size, (1 - self.dropout)):
                    _, step, loss, predication = sess.run(
                        [optimizer, global_step, cost, pred], feed_dict=train)
                    # tags_seqs, _ = utils.viterbi_decode(num, predication, train[self.X_len], trans_matrix)
                    f = self.evaluate(num, predication, train[self.Y])

                    if args.log_tensorboard:
                        summary = sess.run(summary_op, feed_dict={train_loss: loss, train_acc: f})
                        train_summary_writer.add_summary(summary, step)

                    Logger.info('Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, f))

                save_path = saver.save(sess, save_dir + "model.ckpt", global_step=step)
                Logger.info("[+] Model saved in file: %s" % save_path)

                if epoch % self.display_step == 0:
                    rd, loss, acc = 0, 0., 0.
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size):
                        _loss, predication = sess.run(
                            [cost, pred], feed_dict=valid)
                        loss += _loss
                        # tags_seqs, _ = utils.viterbi_decode(num, predication, valid[self.X_len], trans_matrix)
                        f = self.evaluate(num, predication, valid[self.Y])
                        acc += f
                        rd += 1

                    loss /= rd
                    acc /= rd
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                        
                    if args.log_tensorboard:
                        summary = sess.run(summary_valid, feed_dict = {
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)

                    Logger.info('----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
                    Logger.info('Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}'.format(step, loss, acc))
                    Logger.info('Epoch {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter))
            Logger.info('Optimization Finished!')

            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, self.batch_size, shuffle=False):
                _loss, predication = sess.run(
                    [cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                # tags_seqs, _ = utils.viterbi_decode(num, predication, test[self.X_len], trans_matrix)
                f = self.evaluate(num, predication, test[self.Y])
                acc += f
                pred_test_y.extend(predication)
            acc /= rd
            loss /= rd
            return pred_test_y, loss, acc


    def get_batch_data(self, x, y, l, batch_size, keep_prob = 1.0, shuffle = True):
        for index in utils.batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                # self.X_len: l[index],
                self.keep_prob: keep_prob,
            }

            yield feed_dict, len(index)


    def evaluate(self, num, labels, y):
        golds = []
        preds = []
        for i in xrange(num):
            # p_len = y_lens[i]
            golds.append(np.argmax(y[i]))
            preds.append(np.argmax(labels[i]))
        # p = metrics.precision_score(golds, preds, average='macro')
        # r = metrics.recall_score(golds, preds, average='macro')
        # f = metrics.f1_score(golds, preds, average='macro')
        # return (p, r, f)
        # return metrics.f1_score(golds, preds, average='micro')
        return metrics.accuracy_score(golds, preds)
