#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import tensorflow as tf
import config_parser
from utils import makedirs

class ArgsParser(object):
    def __init__(self):
        self.flags = tf.app.flags.FLAGS

    def parse_args(self):
        # 读取配置中的参数，若用户从命令行中指定了相同参数，将会覆盖配置中的参数，优先使用命令行参数
        config = config_parser.ConfigParser()

        # 解析命令行中的参数
        tf.app.flags.DEFINE_string("flag_string", 'yes', 'input a string')
        tf.app.flags.DEFINE_string("template_file", config.template_file, "temple file")
        tf.app.flags.DEFINE_string("word2id", "", "word2id")
        tf.app.flags.DEFINE_string("feature2id", "", "feature2id")
        tf.app.flags.DEFINE_string("label2id", "", "label2id")
        tf.app.flags.DEFINE_string("embedding_file", config.embedding_file, "embedding file")
        tf.app.flags.DEFINE_string("model_type", config.get("general", "model_type"), "model type")
        tf.app.flags.DEFINE_string("output_dir", config.output_dir, "output dir")
        tf.app.flags.DEFINE_string('model_dir', config.model_dir, 'models dir')
        tf.app.flags.DEFINE_string('log_dir_', config.log_dir_, 'tensor board log dir')
        tf.app.flags.DEFINE_string('train_data', config.train_data, 'train data')
        tf.app.flags.DEFINE_string('test_data', config.test_data, 'test data')
        tf.app.flags.DEFINE_string('valid_data', config.valid_data, 'valid data')
        tf.app.flags.DEFINE_string('restore_model', config.restore_model, 'Path of the model to restored')
        tf.app.flags.DEFINE_string('run_type', config.run_type, 'run type contains: train; interaction; export2pb')
        tf.app.flags.DEFINE_string('dics_path', config.dics_path, 'dics file path')
        tf.app.flags.DEFINE_string('conv_filter_size_list', config.get("hyperparameter", "conv_filter_size_list"), 'conv filter size list')

        tf.app.flags.DEFINE_integer("number_words", 15, "word size")
        tf.app.flags.DEFINE_integer("in_dim", 16, "dim size")
        tf.app.flags.DEFINE_integer("hidden_dim", config.get("hyperparameter", "hidden_dim"), "hidden unit number")
        tf.app.flags.DEFINE_integer("num_classes", config.get("hyperparameter", "num_classes"), "tagset size")
        tf.app.flags.DEFINE_integer("batch_size", config.get("hyperparameter", "batch_size"), "number of examples per mini batch")
        tf.app.flags.DEFINE_integer("max_length", config.get("hyperparameter", "max_length"), "max num of tokens per query")
        tf.app.flags.DEFINE_integer("train_steps", config.get("hyperparameter", "train_steps"), "train_steps") # config.get("hyperparameter", "train_steps")
        tf.app.flags.DEFINE_integer('flag_int', 400, 'input a int')
        tf.app.flags.DEFINE_integer("display_step", 1, "number of test display step")
        tf.app.flags.DEFINE_integer("embedding_dim", config.get("data", "embedding_dim"), "embedding dim")
        tf.app.flags.DEFINE_integer("feature_window_size", 1, "feature window size")
        tf.app.flags.DEFINE_integer("num_conv_filters", config.get("hyperparameter", "num_conv_filters"), "num of conv_filters")
        tf.app.flags.DEFINE_integer("num_dic_feature_classes", 0, "num of dic feature classes")

        tf.app.flags.DEFINE_boolean('evaluate_test_flag', True, 'whether evaluate the test data.')
        tf.app.flags.DEFINE_boolean('flag_bool', True, 'input a bool')
        tf.app.flags.DEFINE_boolean("fine_tune_flag", config.get("hyperparameter", "fine_tune_flag"), "whether fine-tuning the embeddings")
        tf.app.flags.DEFINE_boolean('log_tensorboard', True, 'Whether to record the TensorBoard log.')
        tf.app.flags.DEFINE_boolean('use_dics_flag', config.get("general", "use_dics_flag"), 'Whether to use the external dics feature.')

        tf.app.flags.DEFINE_float('flag_float', 0.01, 'input a float')
        tf.app.flags.DEFINE_float("l2_reg", config.get("hyperparameter", "l2_reg"), "L2 regularization weight")
        tf.app.flags.DEFINE_float("lr", config.get("hyperparameter", "lr"), "learning rate")
        tf.app.flags.DEFINE_float("dropout", config.get("hyperparameter", "dropout"), "dropout rate of input layer")

        # 创建参数中的目录
        if self.flags.run_type == "train":
            makedirs(self.flags.output_dir)
            makedirs(self.flags.model_dir)

        makedirs(self.flags.log_dir_)

        # print self.flags.feature2id

        return self.flags


# def makedirs(abs_dir):
#     if os.path.exists(abs_dir) == False:
#         os.makedirs(abs_dir)