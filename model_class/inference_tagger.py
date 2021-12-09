#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import codecs

class InferenceTagger(object):
    def __init__(self, restore_model, model_type = "ckpt"):
        if model_type == "ckpt":
            meta_file = restore_model + ".meta"

            if os.path.exists(meta_file) == False:
                raise "restore model is not set"

            saver = tf.train.import_meta_graph(meta_file)
            self.__session = tf.Session()
            saver.restore(self.__session, restore_model)

            self.__graph = tf.get_default_graph()
        elif model_type == "pb":
            if os.path.exists(restore_model) == False:
                raise "restore model is not set"

            # parse the graph_def file
            self.__session = tf.Session()
            tf.saved_model.loader.load(self.__session, tf.saved_model.tag_constants.SERVING, restore_model)
            self.__graph = tf.get_default_graph()

            # tensor_name_list = [tensor.name for tensor in self.__graph.as_graph_def().node]
            # with codecs.open("/Users/tony/test/tensor_name_2", 'w', 'utf-8') as out:
            #     for tensor_name in tensor_name_list:
            #         print(tensor_name,'\n')
            #         out.write("%s\n" % tensor_name.encode("utf-8"))
        
        self.build()


    @property
    def session(self):
        return self.__session


    @property
    def graph(self):
        return self.__graph


    # build 函数中主要获取 input_placeholder 和 output_res
    def build(self):
        pass


    def run(self, sentence):
        pass