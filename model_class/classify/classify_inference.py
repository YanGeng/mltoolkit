#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import tensorflow as tf
import model_class.inference_tagger as inference_tagger
import utils.utils as utils


class ClassifyInference(inference_tagger.InferenceTagger):
    def build(self):
        # 获取result
        self.__inference_res = self.graph.get_tensor_by_name("line_transform/inference_result:0")
        # self.__transitions = self.graph.get_tensor_by_name("transitions:0")
        
        # 获取输入placeholder
        self.__X_placeholder = self.graph.get_tensor_by_name("inputs/X_placeholder:0")
        # self.__X_len_placeholder = self.graph.get_tensor_by_name("inputs/X_len_placeholder:0")
        self.__output_dropout = self.graph.get_tensor_by_name("inputs/output_dropout:0")


    def run(self, input_data_set, sentences_length = None):
        feature_list = input_data_set.feature_list

        dicts = {
            self.__X_placeholder: feature_list[[0]],
            # self.Y: [],
            # self.__X_len_placeholder: sentences_length[[0]],
            self.__output_dropout: 1.0,
        }

        predication = self.session.run([self.__inference_res], dicts)
        # tags_seqs, _ = utils.viterbi_decode(len(sentences), predication, sentences_length, trans_matrix1)

        # pred_y = []
        # pred_y.extend(tags_seqs)

        return predication
