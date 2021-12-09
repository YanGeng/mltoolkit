#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import ner_tagger

class BI_LSTM_CRF(ner_tagger.NERTagger):
    def special_build(self):
        with tf.name_scope('weigths'):
            self.W = tf.get_variable(shape = [self.hidden_dim * 2, self.num_classes],
                                    initializer = tf.truncated_normal_initializer(stddev=0.01), name='weights')

            self.lstm_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            self.lstm_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.num_classes], name="bias"))

    
    def inference(self, X, X_len, reused = None):
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)
        word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, self.feature_window_size * self.embedding_dim])
        # word_vectors = tf.concat([word_vectors, DIC], -1)

        with tf.variable_scope("label_inference", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_fw,
                cell_bw = self.lstm_bw,
                inputs = word_vectors,
                dtype = tf.float32,
                sequence_length = X_len
            )

            outputs = tf.concat([outputs[0], outputs[1]], 2)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            scores = tf.nn.softmax(scores)
            inference_result = tf.reshape(scores, [-1, self.time_step_size, self.num_classes], name = "inference_result")

            return inference_result


    def inference_with_dic(self, X, DIC, X_len, reused = None):
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)
        word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, self.feature_window_size * self.embedding_dim])
        word_vectors = tf.concat([word_vectors, DIC], -1)

        with tf.variable_scope("label_inference", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_fw,
                cell_bw = self.lstm_bw,
                inputs = word_vectors,
                dtype = tf.float32,
                sequence_length = X_len
            )

            outputs = tf.concat([outputs[0], outputs[1]], 2)
            outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(outputs, self.W), self.b)
            scores = tf.nn.softmax(scores)
            inference_result = tf.reshape(scores, [-1, self.time_step_size, self.num_classes], name = "inference_result")

            return inference_result