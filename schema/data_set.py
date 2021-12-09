#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
p = sys.path
sys.path.append("..")
# sys.path.append("/Users/eleme/Sandbox-Self/MLT")
# sys.path.append("..")
reload(sys)
sys.setdefaultencoding("utf-8")
import codecs
import numpy as np
import utils.parameters as parameters
from sklearn import preprocessing
from collections import defaultdict

class Dictionary(object):
    def __init__(self):
        self.__dics = {}
    
    @property
    def dics(self):
        return self.__dics

    @dics.setter
    def dics(self, dics):
        self.__dics = dics

    def load_dics_from(self, file_dir):
        load_success = False
        if file_dir is not None and os.path.isdir(file_dir):
            for root, _, files in os.walk(file_dir):
                for file_ in files:
                    dic_name = file_.split(".")[0].split("_")[0]
                    full_file = os.path.join(root, file_)
                    # 读取dic文件
                    if self.dics.has_key(dic_name) == False:
                        self.dics[dic_name] = defaultdict(int)

                    with codecs.open(full_file, "r", "utf-8") as in_:
                        for line in in_:
                            self.dics[dic_name][line.rstrip()] += 1

            if self.dics is not None and len(self.dics) > 0:
                load_success = True

        return load_success


# 一行文本即为一个DataNode
class DataNode(object):
    def __init__(self):
        # 字or分词 维度成员变量
        self.__words = []
        self.__word_features = []
        self.__word_labels = []
        self.__word_dic_features = []

        # 整句维度成员变量
        self.__length = None
        self.__sentence_label = None

    @property
    def length(self):
        if self.__length is None:
            self.__length =  len(self.words)

        return self.__length

    @length.setter
    def length(self, length):
        self.__length = length

    @property
    def words(self):
        return self.__words

    @words.setter
    def words(self, words):
        if not (isinstance(words, list) or isinstance(words, np.ndarray)):
            raise ValueError('words must be a list!')

        self.__words = words

    @property
    def word_features(self):
        return self.__word_features

    @word_features.setter
    def word_features(self, word_features):
        if not (isinstance(word_features, list) or isinstance(word_features, np.ndarray)):
            raise ValueError('word_features must be a list!')

        self.__word_features = word_features

    @property
    def word_dic_features(self):
        return self.__word_dic_features

    @word_dic_features.setter
    def word_dic_features(self, word_dic_features):
        if not (isinstance(word_dic_features, list) or isinstance(word_dic_features, np.ndarray)):
            raise ValueError('word_dic_features must be a list!')

        self.__word_dic_features = word_dic_features

    @property
    def word_labels(self):
        return self.__word_labels

    @word_labels.setter
    def word_labels(self, word_labels):
        if not (isinstance(word_labels, list) or isinstance(word_labels, np.ndarray)):
            raise ValueError('word_labels must be a list!')

        self.__word_labels = word_labels

    @property
    def sentence_label(self):
        return self.__sentence_label

    @sentence_label.setter
    def sentence_label(self, sentence_label):
        self.__sentence_label = sentence_label

    def build_dic_features(self, dics):
        if dics is None or dics.dics is None or len(dics.dics) < 1:
            return

        dic_features = [[] for _ in range(len(self.words))]
        sentence = "".join(self.words)
        for (k, value_list) in dics.dics.items():
            for (value, _) in value_list.items():
                start = 0
                end = len(sentence)
                index = sentence.find(value, start, end)
                while (index != -1):
                    value_length =  len(value)
                    for i in range(len(sentence)):
                        if i == index:
                            dic_features[i].append("B-{}".format(k))
                        elif i > index and i < index + value_length:
                            dic_features[i].append("I-{}".format(k))
                        else:
                            # use "OOV", do not use "O", cause we will encode OOV to 0
                            dic_features[i].append(parameters.OOV)
                    start = index + len(value)
                    index = sentence.find(value, start, end)

        self.word_dic_features = dic_features if len(dic_features) > 0 and len(dic_features[0]) > 0 else [[parameters.OOV] for _ in range(len(self.words))]


class OriginalDataSet(object):
    def __init__(self, data_corpus = None, data_lengths = None, max_length = None, type_ = "NER"):
        # data_corpus 为list类型的DataNode结构
        self.__data_corpus = data_corpus if (isinstance(data_corpus, list) or isinstance(data_corpus, np.ndarray)) else None

        self.__data_lengths = data_lengths if (isinstance(data_lengths, list) or isinstance(data_lengths, np.ndarray)) else None

        self.__max_length = max_length if isinstance(max_length, int) else None

        # __sentence_list大小为：[样本数 * 单样本token数（char数or分词数）]
        self.__sentence_list = []

        # __feature_list大小为：[样本数 * 单样本token数 * 单token feature数]
        self.__feature_list = []

        # __dic_feature_list大小为：[样本数 * 单样本token数 * 匹配到的dic数]
        self.__dic_feature_list = []

        # __word_label_list大小为：[样本数 * 单样本token数], 使用于NER
        self.__word_label_list = []

        # __sentence_label_list大小为：[样本数], 使用于classify
        self.__sentence_label_list = []

        self.build(type_)

    @property
    def data_corpus(self):
        return self.__data_corpus

    @data_corpus.setter
    def data_corpus(self, data_corpus):
        if not (isinstance(data_corpus, list) or isinstance(data_corpus, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__word2id = data_corpus

    @property
    def data_lengths(self):
        return self.__data_lengths

    @data_lengths.setter
    def data_lengths(self, data_lengths):
        if not (isinstance(data_lengths, list) or isinstance(data_lengths, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__data_lengths = data_lengths

    @property
    def max_length(self):
        return self.__max_length

    @max_length.setter
    def max_length(self, max_length):
        if not isinstance(max_length, int):
            raise ValueError('score must be a list!')

        self.__max_length = max_length

    @property
    def sentence_list(self):
        return self.__sentence_list

    @property
    def feature_list(self):
        return self.__feature_list

    @property
    def dic_feature_list(self):
        return self.__dic_feature_list

    @property
    def word_label_list(self):
        return self.__word_label_list

    @property
    def sentence_label_list(self):
        return self.__sentence_label_list

    def build(self, type_):
        # for sent in self.__data_corpus:
        #     if type_ == "NER":
        #         self.__sentence_list.append([item['x'] for item in sent])
        #         self.__feature_list.append([item['F'] for item in sent])
        #         self.__word_label_list.append([item['y'] for item in sent])
        #     elif type_ == "CLASSIFY":
        #         X = []
        #         for x in sent[0]["x"]:
        #             X.append([x])
        #         self.__sentence_list.append(X)
        #         self.__feature_list.append([item['F'] for item in sent])
        #         self.__word_label_list.append([item['y'] for item in sent])
        for node in self.__data_corpus:
            if node.words is not None and len(node.words) > 0:
                self.sentence_list.append(node.words)

            if node.word_features is not None and len(node.word_features) > 0:
                self.feature_list.append(node.word_features)

            if node.word_dic_features is not None and len(node.word_dic_features) > 0:
                self.dic_feature_list.append(node.word_dic_features)

            if node.word_labels is not None and len(node.word_labels) > 0:
                self.word_label_list.append(node.word_labels)

            if node.sentence_label is not None:
                self.sentence_label_list.append(node.sentence_label)


class UpdatedDataSet(object):
    # UpdatedDataSet中的变量均转化成了：id（非embedding）
    def __init__(self, sentence_list = None, feature_list = None, word_label_list = None, sentence_label_list = None, sentence_lengths = None
                , max_length = None, sentence_num_classes = None, dic_feature_list = None, word_num_classes = None, dic_num_classed = None):
        # __sentence_list大小为：[样本数 * max_length]
        self.__sentence_list = sentence_list if (isinstance(sentence_list, list) or isinstance(sentence_list, np.ndarray)) else None

        # __feature_list大小为：[样本数 * max_length * 单token feature数]
        self.__feature_list = feature_list if (isinstance(feature_list, list) or isinstance(feature_list, np.ndarray)) else None

        # __word_label_list大小为：[样本数 * max_length]
        self.__word_label_list = word_label_list if (isinstance(word_label_list, list) or isinstance(word_label_list, np.ndarray)) else None

        # __sentence_label_list大小为：[样本数]
        self.__sentence_label_list = sentence_label_list if (isinstance(sentence_label_list, list) or isinstance(sentence_label_list, np.ndarray)) else None

        # __sentence_label_one_hot大小为：[样本数 * sentence_num_classes or (sentence_num_classes + 1)]
        self.__sentence_label_one_hot = None

        # __dic_feature_list大小为：[样本数]
        self.__dic_feature_list = dic_feature_list if (isinstance(dic_feature_list, list) or isinstance(dic_feature_list, np.ndarray)) else None

        # __dic_feature_one_hot大小为：[样本数 * max_length * word_num_classes or (word_num_classes + 1)]
        self.__dic_feature_one_hot = None

        # __sentence_lengths大小为：[样本数]
        self.__sentence_lengths = sentence_lengths if (isinstance(sentence_lengths, list) or isinstance(sentence_lengths, np.ndarray)) else None

        self.__max_length = max_length if isinstance(max_length, int) else None

        self.__sentence_num_classes = sentence_num_classes if isinstance(sentence_num_classes, int) else None

        self.__word_num_classes = word_num_classes if isinstance(word_num_classes, int) else None

        self.__dic_num_classed = dic_num_classed if isinstance(dic_num_classed, int) else None

        self.build()

    @property
    def sentence_list(self):
        return self.__sentence_list

    @sentence_list.setter
    def sentence_list(self, sentence_list):
        if not (isinstance(sentence_list, list) or isinstance(sentence_list, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__sentence_list = sentence_list

    @property
    def feature_list(self):
        return self.__feature_list

    @feature_list.setter
    def feature_list(self, feature_list):
        if not (isinstance(feature_list, list) or isinstance(feature_list, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__feature_list = feature_list

    @property
    def word_label_list(self):
        return self.__word_label_list

    @word_label_list.setter
    def word_label_list(self, word_label_list):
        if not (isinstance(word_label_list, list) or isinstance(word_label_list, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__word_label_list = word_label_list

    @property
    def sentence_label_list(self):
        return self.__sentence_label_list

    @sentence_label_list.setter
    def sentence_label_list(self, sentence_label_list):
        if not (isinstance(sentence_label_list, list) or isinstance(sentence_label_list, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__sentence_label_list = sentence_label_list

    @property
    def sentence_label_one_hot(self):
        return self.__sentence_label_one_hot

    @sentence_label_one_hot.setter
    def sentence_label_one_hot(self, sentence_label_one_hot):
        if not (isinstance(sentence_label_one_hot, list) or isinstance(sentence_label_one_hot, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__sentence_label_one_hot = sentence_label_one_hot

    @property
    def dic_feature_one_hot(self):
        return self.__dic_feature_one_hot

    @dic_feature_one_hot.setter
    def dic_feature_one_hot(self, dic_feature_one_hot):
        if not (isinstance(dic_feature_one_hot, list) or isinstance(dic_feature_one_hot, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__dic_feature_one_hot = dic_feature_one_hot

    @property
    def sentence_lengths(self):
        return self.__sentence_lengths

    @sentence_lengths.setter
    def sentence_lengths(self, sentence_lengths):
        if not (isinstance(sentence_lengths, list) or isinstance(sentence_lengths, np.ndarray)):
            raise ValueError('score must be a list!')

        self.__sentence_lengths = sentence_lengths

    @property
    def max_length(self):
        return self.__max_length

    @max_length.setter
    def max_length(self, max_length):
        if not isinstance(max_length, int):
            raise ValueError('score must be a list!')

        self.__max_length = max_length

    @property
    def sentence_num_classes(self):
        return self.__sentence_num_classes

    @sentence_num_classes.setter
    def sentence_num_classes(self, sentence_num_classes):
        if not isinstance(sentence_num_classes, int):
            raise ValueError('score must be a int!')

        self.__sentence_num_classes = sentence_num_classes

    @property
    def word_num_classes(self):
        return self.__word_num_classes

    @word_num_classes.setter
    def word_num_classes(self, word_num_classes):
        if not isinstance(word_num_classes, int):
            raise ValueError('score must be a int!')

        self.__word_num_classes = word_num_classes

    def build(self):
        # build one-hot sentence label
        if self.sentence_label_list is not None and len(self.sentence_label_list) > 0:
            onehot_encoder = preprocessing.OneHotEncoder(sparse = False, categories = [range(self.sentence_num_classes + 1)])
            sentence_label_2d = [[item] for item in self.sentence_label_list]
            self.sentence_label_one_hot = onehot_encoder.fit_transform(sentence_label_2d)

        # build one-hot dic feature
        if self.__dic_feature_list is not None and len(self.__dic_feature_list) > 0:
            onehot_encoder = preprocessing.OneHotEncoder(sparse = False, categories = [range(self.__dic_num_classed)])
            dic_features_2d = []
            for dic_feature in self.__dic_feature_list:
                new_dic_feature = []
                for token_dic_feature in dic_feature:
                    dic_feature_2d = [[item] for item in token_dic_feature]
                    dic_feature_2d_one_hot = onehot_encoder.fit_transform(dic_feature_2d)
                    # 按行相加，将所有dic_feature相加
                    sum_one_hot = np.sum(dic_feature_2d_one_hot, axis=0)  
                    new_dic_feature.append(sum_one_hot)
                dic_features_2d.append(new_dic_feature)
            self.dic_feature_one_hot = np.array(dic_features_2d)
        # else:
        #     new_dic_features = []
        #     for i in range(len(self.sentence_list)):
        #         new_dic_feature = [[] for _ in range(self.max_length)]
        #         new_dic_features.append(new_dic_feature)
        #     self.dic_feature_one_hot = np.array(new_dic_features)


if __name__ == '__main__':
    dics = Dictionary()
    dics.load_dics_from("/Users/tony/Sandbox-Self/MLT/data/dics")