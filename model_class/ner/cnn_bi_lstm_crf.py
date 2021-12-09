#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import model_class.neural_tagger as neural_tagger
import ner_tagger

# class CNN_BI_LSTM_CRF(neural_tagger.NeuralTagger):
class CNN_BI_LSTM_CRF(ner_tagger.NERTagger):
    def special_build(self):
        with tf.name_scope("weights"):
            self.W = tf.get_variable(shape = [self.hidden_dim * 2, self.num_classes],
                                    initializer = tf.truncated_normal_initializer(stddev = 0.01), name = "weights")
            
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

            # self.conv_weight = tf.get_variable(shape = [2, self.embedding_dim, 1, self.embedding_dim],
            #                         initializer = tf.truncated_normal_initializer(stddev = 0.01), name = "con_weights")
            self.conv_weight = tf.get_variable(shape = [2, 1, self.feature_window_size * self.embedding_dim + self.dic_dim, self.num_conv_filters],
                                    initializer = tf.truncated_normal_initializer(stddev = 0.01), name = "con_weights")

        with tf.name_scope("biases"):
            self.b = tf.Variable(tf.random_uniform([self.num_classes], -0.01, 0.01), name = "bias")
            self.conv_b = tf.Variable(tf.random_uniform([self.num_conv_filters], -0.01, 0.01), name = "conv_bias")


    def inference(self, X, X_len):
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        # word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)

        with tf.variable_scope("convolution"):
            # word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, self.feature_window_size * self.embedding_dim])
            # word_vectors = tf.concat([word_vectors, DIC], -1)
            word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, 1, self.feature_window_size * self.embedding_dim + self.dic_dim])
            
            conv = tf.nn.conv2d(word_vectors, self.conv_weight, strides = [1, 1, 1, 1], padding = "SAME")

            conv = tf.add(conv, self.conv_b)
            conv = tf.reshape(conv, [-1, self.time_step_size, self.num_conv_filters])
            # word_vectors = tf.concat([word_vectors, conv], 2)

        with tf.variable_scope("label_inference"):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_fw,
                cell_bw = self.lstm_bw,
                inputs = conv,
                dtype = tf.float32,
                sequence_length = X_len
            )

            outputs = tf.concat([outputs[0], outputs[1]], 2)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            inference_result = tf.reshape(scores, [-1, self.time_step_size, self.num_classes], name = "inference_result")

        return inference_result


    def inference_with_dic(self, X, DIC, X_len):
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        # word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)

        with tf.variable_scope("convolution"):
            word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, self.feature_window_size * self.embedding_dim])
            word_vectors = tf.concat([word_vectors, DIC], -1)
            word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, 1, self.feature_window_size * self.embedding_dim + self.dic_dim])
            
            conv = tf.nn.conv2d(word_vectors, self.conv_weight, strides = [1, 1, 1, 1], padding = "SAME")

            conv = tf.add(conv, self.conv_b)
            conv = tf.reshape(conv, [-1, self.time_step_size, self.num_conv_filters])
            # word_vectors = tf.concat([word_vectors, conv], 2)

        with tf.variable_scope("label_inference"):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_fw,
                cell_bw = self.lstm_bw,
                inputs = conv,
                dtype = tf.float32,
                sequence_length = X_len
            )

            outputs = tf.concat([outputs[0], outputs[1]], 2)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            inference_result = tf.reshape(scores, [-1, self.time_step_size, self.num_classes], name = "inference_result")

        return inference_result


    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            regularization = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b) + tf.nn.l2_loss(self.conv_weight) + tf.nn.l2_loss(self.conv_b)

            if self.fine_tune_flag:
                regularization += tf.nn.l2_loss(self.embedding_matrix)

            cost += regularization * self.l2_reg
            return cost