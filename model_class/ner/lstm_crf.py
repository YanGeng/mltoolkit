#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import ner_tagger

class LSTM_CRF(ner_tagger.NERTagger):
    def special_build(self):
        with tf.name_scope("weights"):
            self.W = tf.get_variable(shape = [self.hidden_dim, self.num_classes], 
                                    initializer = tf.truncated_normal_initializer(stddev = 0.01), name = "weights")
            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope("biases"):
            self.b = tf.Variable(tf.zeros(self.num_classes), name = "bias")


    def inference(self, X, X_len):
        real_time_step_size = tf.shape(X)[1] # self.time_step_size
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)
        word_vectors = tf.reshape(word_vectors, [-1, real_time_step_size, self.feature_window_size * self.embedding_dim])
        # word_vectors = tf.concat([word_vectors, DIC], -1)

        with tf.variable_scope("label_inference", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.dynamic_rnn(self.lstm_fw, word_vectors, dtype = tf.float32, sequence_length = X_len)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim])
            # outputs = tf.nn.dropout(outputs, keep_prob = self.keep_prob)

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            # scores = tf.nn.softmax(scores)
            inference_result = tf.reshape(scores, [-1, real_time_step_size, self.num_classes], name = "inference_result")

            return inference_result


    def inference_with_dic(self, X, DIC, X_len):
        real_time_step_size = tf.shape(X)[1]
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)
        word_vectors = tf.reshape(word_vectors, [-1, real_time_step_size, self.feature_window_size * self.embedding_dim])
        word_vectors = tf.concat([word_vectors, DIC], -1)

        with tf.variable_scope("label_inference", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.dynamic_rnn(self.lstm_fw, word_vectors, dtype = tf.float32, sequence_length = X_len)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim])
            # outputs = tf.nn.dropout(outputs, keep_prob = self.keep_prob)

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            # scores = tf.nn.softmax(scores)
            inference_result = tf.reshape(scores, [-1, real_time_step_size, self.num_classes], name = "inference_result")

            return inference_result