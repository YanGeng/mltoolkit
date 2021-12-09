#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import tensorflow as tf
import model_class.inference_tagger as inference_tagger
import utils.utils as utils


class NERInference(inference_tagger.InferenceTagger):
    def build(self):
        # 获取result
        self.__inference_res = self.graph.get_tensor_by_name("line_transform/inference_result:0")
        self.__transitions = self.graph.get_tensor_by_name("transitions:0")
        
        # 获取输入placeholder
        self.__X_placeholder = self.graph.get_tensor_by_name("inputs/X_placeholder:0")
        self.__Dics_placeholder = self.graph.get_tensor_by_name("inputs/D_placeholder:0")
        self.__X_len_placeholder = self.graph.get_tensor_by_name("inputs/X_len_placeholder:0")
        self.__output_dropout = self.graph.get_tensor_by_name("inputs/output_dropout:0")


    def run(self, input_data_set, use_dics_flag = False):
        feature_list = input_data_set.feature_list
        sentences_length = input_data_set.sentence_lengths
        dic_feature = input_data_set.dic_feature_one_hot

        if use_dics_flag:
            dicts = {
                self.__X_placeholder: feature_list[[0]],
                self.__Dics_placeholder : dic_feature[[0]],
                self.__X_len_placeholder: sentences_length[[0]],
                self.__output_dropout: 1.0,
            }
        else:
            dicts = {
                self.__X_placeholder: feature_list[[0]],
                # self.Y: [],
                self.__X_len_placeholder: sentences_length[[0]],
                self.__output_dropout: 1.0,
            }

        trans_matrix1, predication = self.session.run([self.__transitions, self.__inference_res], dicts)
        tags_seqs, _ = utils.viterbi_decode(len(feature_list), predication, sentences_length, trans_matrix1)

        pred_y = []
        pred_y.extend(tags_seqs)

        return pred_y
