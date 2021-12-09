#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import codecs
import numpy as np
from utils.utils import read_emb_from_file, makedirs, Logger

class Input2ID(object):
    def __init__(self, word2id = None, feature2id = None, sentence_label2id = None, word_label2id = None, dic_feature2id = None, embedding_file = None):
        self.__word2id = word2id if isinstance(word2id, dict) else None
        self.__feature2id = feature2id if isinstance(feature2id, dict) else None
        self.__sentence_label2id = sentence_label2id if isinstance(sentence_label2id, dict) else None
        self.__word_label2id = word_label2id if isinstance(word_label2id, dict) else None
        self.__dic_feature2id = dic_feature2id if isinstance(dic_feature2id, dict) else None
        self.__id2sentence_label = None
        self.__id2word_label = None
        self.__id2vector = None
        self.__id2line_number = None

        self.build_id2label()
        self.build_id2vector(embedding_file)


    @property
    def word2id(self):
        if self.__word2id is not None and len(self.__word2id) > 0:
            return self.__word2id
        elif self.__feature2id is not None and len(self.__feature2id) > 0:
            return self.__feature2id
        else:
            return None


    @word2id.setter
    def word2id(self, word2id):
        if not isinstance(word2id, dict):
            raise ValueError('score must be a dict!')

        self.__word2id = word2id


    @property
    def feature2id(self):
        if self.__feature2id is not None and len(self.__feature2id) > 0:
            return self.__feature2id
        elif self.__word2id is not None and len(self.__word2id) > 0:
            return self.__word2id
        else:
            return None


    @feature2id.setter
    def feature2id(self, feature2id):
        if not isinstance(feature2id, dict):
            raise ValueError('feature2id must be a dict!')

        self.__feature2id = feature2id


    # 若不指定具体获取哪个label2id，默认返回第一个不为空的（非multitask任务）
    @property
    def label2id(self):
        if self.sentence_label2id is not None and len(self.sentence_label2id) > 0:
            return self.sentence_label2id
        elif self.word_label2id is not None and len(self.word_label2id) > 0:
            return self.word_label2id
        else:
            return None


    @property
    def sentence_label2id(self):
        return self.__sentence_label2id


    @sentence_label2id.setter
    def sentence_label2id(self, sentence_label2id):
        if not isinstance(sentence_label2id, dict):
            raise ValueError('sentence_label2id must be a dict!')

        self.__sentence_label2id = sentence_label2id


    @property
    def word_label2id(self):
        return self.__word_label2id


    @word_label2id.setter
    def word_label2id(self, word_label2id):
        if not isinstance(word_label2id, dict):
            raise ValueError('word_label2id must be a dict!')

        self.__word_label2id = word_label2id


    @property
    def dic_feature2id(self):
        return self.__dic_feature2id


    @dic_feature2id.setter
    def dic_feature2id(self, dic_feature2id):
        if not isinstance(dic_feature2id, dict):
            raise ValueError('dic_feature2id must be a dict!')

        self.__dic_feature2id = dic_feature2id


    @property
    def id2label(self):
        if self.id2sentence_label is not None:
            return self.id2sentence_label
        elif self.id2word_label is not None:
            return self.id2word_label
        else:
            return None


    @property
    def id2sentence_label(self):
        return self.__id2sentence_label


    @property
    def id2word_label(self):
        return self.__id2word_label


    @property
    def id2vector(self):
        return self.__id2vector


    def build_id2vector(self, embedding_file):
        if embedding_file is not None:
            self.__id2vector, self.__id2line_number = read_emb_from_file(embedding_file, self.feature2id)


    def build_id2label(self):
        if self.__sentence_label2id is not None and len(self.__sentence_label2id) > 0:
            self.__id2sentence_label = dict((k, v) for v, k in self.__sentence_label2id.iteritems())

        if self.__word_label2id is not None and len(self.__word_label2id) > 0:
            self.__id2word_label = dict((k, v) for v, k in self.__word_label2id.iteritems())


    def export2file(self, export_dir):
        makedirs(export_dir)
        Logger.info("Dicts file is stored in path: {}".format(export_dir))
        if self.id2vector is not None and len(self.id2vector) > 0:
            with codecs.open(export_dir + 'ID2VECTOR', 'w', 'utf-8') as out:
                id = 0
                for values in self.__id2vector:
                    out.write("%s" % id)
                    for value in values:
                        out.write(" %.20f" % value)
                    out.write("\n")
                    out.flush()
                    id += 1

        if self.__id2line_number is not None and len(self.__id2line_number) > 0:
            with codecs.open(export_dir + 'ID2LINENUMBER', 'w', 'utf-8') as out:
                for k, v in self.__id2line_number.iteritems():
                    out.write("%s\t%d\n" % (k, v))

        if self.word2id is not None and len(self.word2id) > 0:
            with codecs.open(export_dir + 'WORDS2ID', 'w', 'utf-8') as out:
                for k, v in self.__word2id.iteritems():
                    out.write("%s\t%d\n" % (k, v))

        if self.sentence_label2id is not None and len(self.sentence_label2id) > 0:
            with codecs.open(export_dir + 'sentence_label2id', 'w', 'utf-8') as out:
                for k, v in self.sentence_label2id.iteritems():
                    out.write("%s\t%d\n" % (k, v))

        if self.word_label2id is not None and len(self.word_label2id) > 1: # word_label2id 默认会加一个 _OOV_
            with codecs.open(export_dir + 'word_label2id', 'w', 'utf-8') as out:
                for k, v in self.word_label2id.iteritems():
                    out.write("%s\t%d\n" % (k, v))

        if self.dic_feature2id is not None and len(self.dic_feature2id) > 1: # dic_feature2id 默认会加一个 _OOV_
            with codecs.open(export_dir + 'dic_feature2id', 'w', 'utf-8') as out:
                for k, v in self.dic_feature2id.iteritems():
                    out.write("%s\t%d\n" % (k, v))


    def load_from_dir(self, load_dir):
        # 读取id2vector文件
        id2vector = []
        with codecs.open(load_dir + "/ID2VECTOR", "r", "utf-8") as in_:
            for line in in_:
                columns = line.rstrip().split(" ")
                if len(columns) < 2:
                    continue

                # id = columns[0]
                vector_str = columns[1:]
                vector_float = [float(v) for v in vector_str]
                id2vector.append(vector_float)

        self.__id2vector = np.array(id2vector)

        # 读取word2id文件
        self.__word2id = {}
        with codecs.open(load_dir + "/WORDS2ID", "r", "utf-8") as in_:
            for line in in_:
                columns = line.rstrip().split("\t")
                if len(columns) < 2:
                    continue

                self.__word2id[columns[0]] = int(columns[1])

        # 读取sentence_label2id文件
        self.sentence_label2id = {}
        self.__id2sentence_label = {}
        try :
            with codecs.open(load_dir + "/sentence_label2id", "r", "utf-8") as in_:
                for line in in_:
                    columns = line.rstrip().split("\t")
                    if len(columns) < 2:
                        continue

                    label = columns[0]
                    id = int(columns[1])
                    self.sentence_label2id[label] = id
                    self.__id2sentence_label[id] = label
        except Exception:
            self.sentence_label2id = None
            self.id2sentence_label = None
            Logger.error("sentence_label2id 不存在")
        else:
            Logger.info("读取sentence_label2id成功")

        # 读取word_label2id文件
        self.word_label2id = {}
        self.__id2word_label = {}
        try :
            with codecs.open(load_dir + "/word_label2id", "r", "utf-8") as in_:
                for line in in_:
                    columns = line.rstrip().split("\t")
                    if len(columns) < 2:
                        continue

                    label = columns[0]
                    id = int(columns[1])
                    self.word_label2id[label] = id
                    self.__id2word_label[id] = label
        except Exception:
            self.word_label2id = None
            self.id2word_label = None
            Logger.error("word_label2id 不存在")
        else:
            Logger.info("读取word_label2id成功")

        # 读取dic_feature2id文件
        self.dic_feature2id = {}
        try :
            with codecs.open(load_dir + "/dic_feature2id", "r", "utf-8") as in_:
                for line in in_:
                    columns = line.rstrip().split("\t")
                    if len(columns) < 2:
                        continue

                    label = columns[0]
                    id = int(columns[1])
                    self.dic_feature2id[label] = id
        except Exception:
            self.dic_feature2id = None
            self.id2word_label = None
            Logger.error("dic_feature2id 不存在")
        else:
            Logger.info("读取dic_feature2id成功")


if __name__ == '__main__':
    list1 = [1,2,3]
    dict1 = {1:list1,3:4}
    inputs = Input2ID()
    inputs.word2id = dict1

    input2 = Input2ID(dict1)

    input2.load_from_dir("/Users/tony/Sandbox-Self/MLT/export")