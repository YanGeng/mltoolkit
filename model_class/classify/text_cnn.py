#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
# import model_class.neural_tagger as neural_tagger
import classify_tagger

class Text_CNN(classify_tagger.ClassifyTagger):
    def special_build(self):
        with tf.name_scope("weights"):
            self.W = tf.get_variable(shape = [self.num_conv_filters * len(self.filter_size_list), self.num_classes], 
                                    initializer = tf.truncated_normal_initializer(stddev = 0.01), name = "weights")

        with tf.name_scope("biases"):
            self.b = tf.Variable(tf.zeros(self.num_classes), name = "bias")


    def inference(self, X, X_len):
        word_vectors = tf.nn.embedding_lookup(self.embedding_matrix, X)
        # word_vectors = tf.nn.dropout(word_vectors, keep_prob = self.keep_prob)
        # word_vectors = tf.reshape(word_vectors, [-1, self.time_step_size, self.feature_window_size * self.embedding_dim])
        # word_vectors = tf.expand_dims(word_vectors, -1)

        pooled_outputs = []
        for _, filter_size in enumerate(self.filter_size_list):
            with tf.name_scope("conv_maxpool_{}".format(filter_size)):
                filter_shape = [filter_size, self.feature_window_size, self.embedding_dim, self.num_conv_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.2, name = "W"))
                b = tf.Variable(tf.constant(0.1, shape = [self.num_conv_filters]), name = "b")
                conv = tf.nn.conv2d(word_vectors, W, strides = [1, 1, 1, 1], padding = "VALID", name = "conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")

                pooled = tf.nn.max_pool(h, ksize = [1, self.time_step_size - filter_size + 1, 1, 1], strides = [1, 1, 1, 1],
                                        padding = "VALID", name = "pool")

                pooled_outputs.append(pooled)

        total_filter_size = self.num_conv_filters * len(self.filter_size_list)
        total_pooled_outputs = tf.concat(pooled_outputs, 3)
        total_pooled_outputs = tf.reshape(total_pooled_outputs, [-1, total_filter_size])

        with tf.name_scope("line_transform"):
            scores = tf.add(tf.matmul(total_pooled_outputs, self.W), self.b)
            # inference_result = tf.nn.softmax(scores, name = "inference_result")
            inference_result = tf.argmax(scores, 1, name = "inference_result")

            return scores, inference_result