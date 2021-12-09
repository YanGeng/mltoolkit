#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import codecs
import copy
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn import preprocessing
import parameters
from schema import input2id, data_set
from features import Template, feature_extractor, build_features
from parameters import START, END
from logger import Logger

def pretreatment(train_data, valid_data, test_data, ignore_threshhold = 0, template = None, args = None):
    embedding_file = args.embedding_file
    model_type = args.model_type

    dics = None
    if args is not None and args.use_dics_flag == True:
        dic_path =  args.dics_path
        dics = data_set.Dictionary()
        # 如果没有dic，则将args.use_dics_flag至为false
        args.use_dics_flag = dics.load_dics_from(dic_path)

    train_data_set = read_corpus(train_data, template = template, dics = dics, model_type = model_type)
    valid_data_set = read_corpus(valid_data, template = template, dics = dics, model_type = model_type)
    test_data_set = read_corpus(test_data, template = template, dics = dics, model_type = model_type)

    # 获取实际最大的max_length
    max_length = max(train_data_set.max_length, valid_data_set.max_length, test_data_set.max_length)

    # 根据所有corpus，生成dicts
    # feature_2_id, word_2_id, label_2_id = convert_2_id(train_data_set.data_corpus, valid_data_set.data_corpus, 
    #                                                     test_data_set.data_corpus, ignore_threshhold)
    input2id = convert_2_id(train_data_set, valid_data_set, 
                            test_data_set, ignore_threshhold, embedding_file)

    # id_dict = {"word2id" : word_2_id, "feature2id" : feature_2_id, "label2id" : label_2_id}

    # train = (train_corpus, train_lengths)
    # valid = (valid_corpus, valid_lengths)
    # test = (test_corpus, test_lengths)

    return train_data_set, valid_data_set, test_data_set, input2id, max_length


def read_corpus(input_file, template = None, model_type = None, dics = None, encoding = "utf-8"):
    """
    Reads corpus file, then returns the list of sentences and labelSeq.

    ## Parameters
    template: feature templates instance

    ## Returns
    corpus:  the dataset consist of multiple sent-item
    lengths:  the length of each sentence in corpus
    max_len: the maximum length of sentences
    """
    if input_file is None:
        raise ValueError("input data file is: None")

    assert type(template) == Template

    fields = template.fields
    with codecs.open(input_file, encoding = encoding) as cs:
        max_length = parameters.MAX_LEN
        lengths = []
        dataNode_list = []

        type_ = None
        if (model_type == "LSTM" or model_type == "BILSTM" or model_type == "CNNBILSTM"):
            # sentence之间使用两个换行分隔
            streams = cs.read().strip().split("\n\n")
            for stream in streams:
                # sentence内每个token之前使用一个换行分隔
                tokens = stream.strip().split("\n")
                length = len(tokens)
                # 记录每个corpus中每条数据的长度，及输出token的长度
                lengths.append(length)
                max_length = max(max_length, length)
                sentence = []
                for token in tokens:
                    # 按列分隔
                    columns = token.split()
                    assert len(columns) == len(fields) # 每一列对应于相应的field，他们的length相等

                    # item存储每个token包含的key，value：w：当前token y: token对应的label
                    item = {}
                    for i in range(len(fields)):
                        # fields[i]为key；columns[i]为value
                        item[fields[i]] = columns[i]
                    sentence.append(item)
                # 根据template中定义的Feature Templates抽取contextual features
                # 每句文本一个feature结构，为一个list，list中每个节点结构示例为：key : value，结构，如下（“支持李开复”）：
                # w : 支
                # y : O
                # x : [支]
                # F : [</s>, </s>, 支, 持, 李]
                features = apply_feature_templates(sentence, template = template)
                dataNode = data_set.DataNode()
                for feature in features:
                    dataNode.words.append(feature["w"])
                    dataNode.word_features.append(feature["F"])
                    dataNode.word_labels.append(feature["y"])
                    dataNode.length = len(dataNode.words)

                if dics is not None:
                    dataNode.build_dic_features(dics)

                dataNode_list.append(dataNode)
            type_ = "NER"
        elif (model_type == "TEXTCNN"):
            for line in cs:
                columns = line.rstrip().split("\t")
                dataNode = data_set.DataNode()
                dataNode.words = list(columns[0])
                dataNode.sentence_label = columns[1]
                dataNode.word_features = build_features(dataNode.words, template)
                
                dataNode_list.append(dataNode)
            type_ = "CLASSIFY"
        else:
            Logger.error("Model_type is None!!!")
            raise ValueError("Model_type is None!!!")
        
        # 统计每个corpus中每条数据的长度，及输出token的长度
        lengths = np.asarray(lengths, dtype = np.int32)
        return data_set.OriginalDataSet(dataNode_list, lengths, max_length, type_)


def apply_feature_templates(sentence_token, template = None):
    """
    Apply feature templates, generate features for each sentence.
    """
    if not template:
        raise TypeError("Except a valid Template object but got a \'None\'")

    if template.valid:
        features = feature_extractor(sentence_token, template = template)
    return features


def convert_2_id(train_corpus, valid_corpus, test_corpus, ignore_threshhold, embedding_file = None):
    """
    Returns three dicts, which are feature dictionary, word dictionary
    and label dictionary.

    ## Params
        train: train data corpus
        valid: valid data corpus
        test: test data corpus
        threshold: threshold value of feature frequency
        mode: the type of embeddings(char/charpos)
        anno: whether the data is annotated
    ## Return
        returns 3 dicts
    """

    # 获取corpus中key下的所有值
    # def get_value(corpus, key):
    #     return [[item[key] for item in feature] for feature in corpus]

    # words = (get_value(train_corpus, "x") 
    #         + ([] if valid_corpus is None else get_value(valid_corpus, "x"))
    #         + ([] if test_corpus is None else get_value(test_corpus, "x")))els
    sentence_label_list = None
    if train_corpus.sentence_label_list is not None and len(train_corpus.sentence_label_list) > 0:
        sentence_label_list = (train_corpus.sentence_label_list + ([] if valid_corpus is None else valid_corpus.sentence_label_list)
            + ([] if test_corpus is None else test_corpus.sentence_label_list))

    word_label_list = None
    if train_corpus.word_label_list is not None and len(train_corpus.word_label_list) > 0:
        word_label_list = (train_corpus.word_label_list + ([] if valid_corpus is None else valid_corpus.word_label_list)
            + ([] if test_corpus is None else test_corpus.word_label_list))

    features = (train_corpus.feature_list + ([] if valid_corpus is None else valid_corpus.feature_list)
            + ([] if test_corpus is None else test_corpus.feature_list))

    merged_dic_feature_list = None
    if train_corpus.dic_feature_list is not None and len(train_corpus.dic_feature_list) > 0:
        merged_dic_feature_list = (train_corpus.dic_feature_list + ([] if valid_corpus is None else valid_corpus.dic_feature_list)
            + ([] if test_corpus is None else test_corpus.dic_feature_list))

    # labels = (get_value(train_corpus, "y")
    #         + ([] if valid_corpus is None else get_value(valid_corpus, "y"))
    #         + ([] if test_corpus is None else get_value(test_corpus, "y")))

    # features = (get_value(train_corpus, "F")
    #         + ([] if valid_corpus is None else get_value(valid_corpus, "F"))
    #         + ([] if test_corpus is None else get_value(test_corpus, "F")))

    feature_2_id = None
    if features is not None and len(features) > 0:
        # 统计feature中每个词在data中出现的频次
        feature_2_frequent = defaultdict(int)
        for sentence_feature in features:
            for token_feature in sentence_feature:
                for item in token_feature:
                    feature_2_frequent[item] += 1

        # 词频高于ignore_threshhold的，递增赋值id
        feature_2_id = {parameters.OOV: 0}
        current_id = 1
        for sentence_feature in features:
            for token_feature in sentence_feature:
                for item in token_feature:
                    if item not in feature_2_id and feature_2_frequent[item] > ignore_threshhold:
                        feature_2_id[item] = current_id
                        current_id += 1

    # word与feature数据相同，均为原始的words（feature多了：oov, start, end）
    word_2_id = feature_2_id

    # sentence label to id 递增赋值
    sentence_label_2_id = {}
    if sentence_label_list is not None:
        current_id = 1
        for sentence_label in sentence_label_list:
            if sentence_label not in sentence_label_2_id:
                sentence_label_2_id[sentence_label] = current_id
                current_id += 1

    # word_label_2_id 递增赋值，初始化 OOV：0
    word_label_2_id = {parameters.OOV: 0}
    if word_label_list is not None:
        current_id = 1
        for word_labels in word_label_list:
            for word_label in word_labels:
                if word_label not in word_label_2_id:
                    word_label_2_id[word_label] = current_id
                    current_id += 1

    dic_feature_2_id = None
    if merged_dic_feature_list is not None and len(merged_dic_feature_list) > 0:
        # 计算 dic_feature_2_id 递增赋值，初始化 OOV：0
        dic_feature_frequent = defaultdict(int)
        for merged_dic_feature in merged_dic_feature_list:
            for token_dic_feature_list in merged_dic_feature:
                for token_dic_feature in token_dic_feature_list:
                    dic_feature_frequent[token_dic_feature] += 1

        dic_feature_2_id = copy.deepcopy(word_label_2_id)
        current_id = len(dic_feature_2_id)
        for k in dic_feature_frequent:
            if k not in dic_feature_2_id:
                dic_feature_2_id[k] = current_id
                current_id += 1
    
    return input2id.Input2ID(feature_2_id, word_2_id, sentence_label_2_id, word_label_2_id, dic_feature_2_id, embedding_file)


def unfold_corpus(corpus):
    """
    Unfolds a corpus, converts it's sentences from a list of sent-item into 3
    independent lists, the window-repr words list the labels list and the
    features list.

    # Return
        sentcs: a list of sentences, each sent is a list of window-repr words
        featvs: a list of features list for each sent's each word
        labels: a list of labels' sequences
    """
    sentcs = []
    labels = []
    featvs = []
    for sent in corpus:
        sentcs.append([item['x'] for item in sent])
        labels.append([item['y'] for item in sent])
        featvs.append([item['F'] for item in sent])

    return sentcs, featvs, labels


def convert_corpus(originalDataSet, input2id, max_length = parameters.MAX_LEN):
    """
    Converts the list of sentences and labelSeq. After conversion, it will
    returns a 2D tensor which contains word's numeric id sequences, and a 3D
    tensor which contains the one-hot label vectors. All of these tensors have
    been padded(post 0) by given MAX_LEN.

    The shape of returned 2D tensor is (len(sentcs), MAX_LEN), while the 3D
    tensor's is (len(sentcs), MAX_LEN, len(label2idx) + 1).

    ## Parameters
        sentcs: the list of corpus' sentences.
        featvs: the list of feats for sentences.
        labels: the list of sentcs's label sequences.
        word2idx: the vocabulary of words, a map.
        word2idx: the vocabulary of labels, a map.
        max_len: the maximum length of input sentence.

    ## Returns
    new_sentcs: 3D tensor of input corpus with shape
                (corpus_len, seq_len, window)
    new_featvs: 3D tensor of input features with shape
                (corpus_len, seq_len, templates)
    new_labels: 2D tensor of input labels with shape
                (corpus_len, seq_len)
    """
    sentences = originalDataSet.sentence_list
    features = originalDataSet.feature_list
    dic_features = originalDataSet.dic_feature_list
    word_labels = originalDataSet.word_label_list
    sentence_labels = originalDataSet.sentence_label_list
    data_lengths = originalDataSet.data_lengths

    word2id = input2id.word2id
    feat2id = input2id.feature2id
    word_label2id = input2id.word_label2id
    dic_feature2id = input2id.dic_feature2id
    sentence_label2id = input2id.sentence_label2id

    tmp_sent = []
    tmp_feat = []
    new_dic_features = []
    tmp_word_lab = []
    tmp_sentence_label = []
    for sent, feat in zip(sentences, features):
        sent = [word2id.get(w, 0) for w in sent]
        feat = conv_features(feat, feat2id, max_length)

        tmp_sent.append(sent)
        tmp_feat.append(feat)

    if word_labels is not None and len(word_labels) > 0 and word_label2id is not None and len(word_label2id) > 0:
        for word_label in word_labels:
            word_lab = [word_label2id.get(l, 0) for l in word_label]
            tmp_word_lab.append(word_lab)
        tmp_word_lab = pad_sequences(tmp_word_lab, maxlen = max_length, padding = 'post')

    if sentence_labels is not None and len(sentence_labels) > 0 and sentence_label2id is not None and len(sentence_label2id) > 0:
        for sentence_label in sentence_labels:
            tmp_sentence_label.append(sentence_label2id.get(sentence_label, 0))
        # tmp_sentence_label = pad_sequences(tmp_sentence_label, maxlen = max_length, padding = 'post')

    if dic_features is not None and len(dic_features) > 0 and dic_feature2id is not None and len(dic_feature2id) > 0:
        for dic_feature in dic_features:
            new_dic_feature = []
            sent_len = len(dic_feature)
            feat_num = 0
            for token_dic_feature in dic_feature:
                feat_num = len(token_dic_feature)
                tmp_dic2id = [dic_feature2id.get(l, 0) for l in token_dic_feature]
                new_dic_feature.append(tmp_dic2id)

            for _ in xrange(max_length - sent_len):
                padding = []
                for _ in range(feat_num):
                    padding.append(0)
                new_dic_feature.append(padding)

            new_dic_features.append(new_dic_feature)
    # else:
    #     for i in range(len(sentences)):
    #         new_dic_feature = [[] for _ in range(max_length)]
    #         new_dic_features.append(new_dic_feature)

    tmp_sent = pad_sequences(tmp_sent, maxlen = max_length, padding = 'post')
    tmp_feat = np.array(tmp_feat)
    tmp_sentence_label = np.array(tmp_sentence_label)

    # get num_classes
    sentence_num_classes = len(input2id.sentence_label2id)
    word_num_classes = len(input2id.word_label2id)
    dic_num_classed = len(input2id.dic_feature2id) if input2id is not None and input2id.dic_feature2id is not None else 0

    return data_set.UpdatedDataSet(tmp_sent, tmp_feat, tmp_word_lab, tmp_sentence_label, np.array(data_lengths), max_length
                                    , sentence_num_classes, new_dic_features, word_num_classes, dic_num_classed)


def convert_sentence(X, Y, word2idx, label2idx):
    sentc = [[word2idx.get(w_t, 0) for w_t in w] for w in X]
    label = [label2idx.get(l, 0) for l in Y]
    return sentc, label


def conv_features(F, feat2idx, max_len = parameters.MAX_LEN):
    feat_num = len(F[0])
    sent_len = len(F)
    feats = []
    for feat in F:
        feats.append([feat2idx.get(f, 0) for f in feat])
    for i in xrange(max_len - sent_len):
        feats.append([0] * feat_num)
    return feats


def input_prepare(train_data, valid_data, test_data, ignore_threshhold = 0, template = None, args = None):
    Logger.info("#" * 67)
    Logger.info("# Loading data from:")
    Logger.info("#" * 67)
    Logger.info("Train: %s" % args.train_data)
    Logger.info("Valid: %s" % args.valid_data)
    Logger.info("Test:  %s" % args.test_data)

    original_train_set, original_valid_set, original_test_set, input2id, max_length = pretreatment(train_data, valid_data, test_data
                                                    ,ignore_threshhold = 0, template = template, args = args)

    args.max_length = max_length    # 改逻辑后续调整为：按args.max_length截取data corpus

    feature2id = input2id.feature2id
    word2id = input2id.word2id
    label2id = input2id.label2id
    args.label2id = label2id
    args.word2id = word2id
    args.feature2id = feature2id

    Logger.info("Lexical word size:     %d" % len(feature2id))
    Logger.info("Label size:            %d" % len(label2id))
    Logger.info("-------------------------------------------------------------------")
    Logger.info("Training data size:    %d" % len(original_train_set.data_corpus))
    Logger.info("Validation data size:  %d" % len(original_valid_set.data_corpus))
    Logger.info("Test data size:        %d" % len(original_test_set.data_corpus))
    Logger.info("Maximum sentence len:  %d" % max_length)

    Logger.info("embeddings size: {}".format(input2id.id2vector.shape))
    if args.fine_tune_flag:
        Logger.info("The embeddings will be fine-tuned!")

    return original_train_set, original_valid_set, original_test_set, input2id, max_length


def query_prepare(sentence, input2id, template = None, dics = None, args = None):
    unicode_sentence = unicode(sentence, "utf-8")
    sentence_length = len(unicode_sentence)

    dataNode = data_set.DataNode()
    dataNode.words = list(unicode_sentence)
    dataNode.word_features = build_features(dataNode.words, template)

    dics = None
    if args is not None and args.use_dics_flag == True:
        dic_path =  args.dics_path

        dics = data_set.Dictionary()
        # 如果没有dic，则将args.use_dics_flag至为false
        args.use_dics_flag = dics.load_dics_from(dic_path)

    if dics is not None:
        dataNode.build_dic_features(dics)
    
    dataNode_list = []
    legth_list = []
    dataNode_list.append(dataNode)
    legth_list.append(sentence_length)

    maxLength = sentence_length # args.max_length
    originalDataSet = data_set.OriginalDataSet(dataNode_list, legth_list, maxLength, "ner")
    updated_train_set = convert_corpus(originalDataSet, input2id, max_length = maxLength)

    return updated_train_set


# keras API
pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences