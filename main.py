#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import codecs
import tensorflow as tf
from utils import config_parser, args_parser, features
from utils.pretreatment import convert_corpus, input_prepare, query_prepare
from utils.utils import read_emb_from_file, conv_data, eval_ner, convert_id_to_word, evaluate, write_prediction
from utils.logger import Logger
from model_class.ner import lstm_crf, bi_lstm_crf, cnn_bi_lstm_crf, ner_inference
from model_class.classify import text_cnn, classify_inference
from model_class import neural_tagger
from schema import input2id
from tensorflow.python.framework import graph_util


# %%
args_parser = args_parser.ArgsParser()
args = args_parser.parse_args()


def train():
    Logger.info("Start")
    template = features.Template(args.template_file)
    args.feature_window_size = len(template.feature_template)

    original_train_set, original_valid_set, original_test_set, input2id_, max_length = input_prepare(args.train_data, args.valid_data
                                                            , args.test_data, ignore_threshhold = 0, template = template, args = args)

    id2vector = input2id_.id2vector # read_emb_from_file(args.embedding_file, input2id.feature2id)
    args.embedding_dim = max(id2vector.shape[1], args.embedding_dim) # 获取最大的作为embedding_dim
    args.num_dic_feature_classes = len(input2id_.dic_feature2id) if input2id_ is not None and input2id_.dic_feature2id is not None else 0

    Logger.info("Preparing training, validate and testing data.")
    updated_train_set = convert_corpus(original_train_set, input2id_, max_length = args.max_length)
    updated_valid_set = convert_corpus(original_valid_set, input2id_, max_length = args.max_length)
    updated_test_set = convert_corpus(original_test_set, input2id_, max_length = args.max_length)


    Logger.info("Model type is {}".format(args.model_type))
    if args.model_type.upper() == 'LSTM':
        model_type = lstm_crf.LSTM_CRF
        args.num_classes = len(input2id_.word_label2id) + 1
    elif args.model_type.upper() == "BILSTM":
        model_type = bi_lstm_crf.BI_LSTM_CRF
        args.num_classes = len(input2id_.word_label2id) + 1
    elif args.model_type.upper() == "CNNBILSTM":
        model_type = cnn_bi_lstm_crf.CNN_BI_LSTM_CRF
        args.num_classes = len(input2id_.word_label2id) + 1
    elif args.model_type.upper() == "TEXTCNN":
        model_type = text_cnn.Text_CNN
        args.num_classes = len(input2id_.sentence_label2id) + 1
    else:
        raise TypeError("Unknow model type {}".format(args.model_type))

    # log some paremeters
    Logger.info("#" * 67)
    Logger.info("Training arguments")
    Logger.info("#" * 67)
    Logger.info("L2 regular:    %f" % args.l2_reg)
    Logger.info("nb_classes:    %d" % args.num_classes)
    Logger.info("Batch size:    %d" % args.batch_size)
    Logger.info("Hidden layer:  %d" % args.hidden_dim)
    Logger.info("Train epochs:  %d" % args.train_steps)
    Logger.info("Learning rate: %f" % args.lr)

    Logger.info("#" * 67)
    Logger.info("Training process start.")
    Logger.info("#" * 67)

    # build model
    model = model_type(id2vector, args)

    pred_test, test_loss, test_acc = model.run(updated_train_set, updated_valid_set, updated_test_set, args)

    Logger.info("Test loss: %f, accuracy: %f" % (test_loss, test_acc))
    # comment it, these logical need to be refine
    # pred_test = [pred_test[i][:original_test_set.data_lengths[i]] for i in xrange(len(pred_test))]
    # pred_test_label = convert_id_to_word(pred_test, input2id_.id2label)
    # if args.evaluate_test_flag:
    #     res_test, pred_test_label = evaluate(pred_test_label, original_test_set.word_label_list)
    #     Logger.info("Test F1: %f, P: %f, R: %f" % (res_test['f1'], res_test['p'], res_test['r']))

    # original_text = original_test_set.sentence_list # [[item['w'] for item in sent] for sent in original_test_set.data_corpus]
    # write_prediction(args.output_dir + 'prediction.utf8', original_text, pred_test_label)

    Logger.info("Saving feature dicts...")
    input2id_.export2file(args.output_dir)

    Logger.info("The end")


def interaction():
    # 加载相关模型id
    input2id_ = input2id.Input2ID()
    input2id_.load_from_dir(args.output_dir)

    args.embedding_dim = max(input2id_.id2vector.shape[1], args.embedding_dim)

    print args.model_type
    if args.model_type == "LSTM":
        inference_type = ner_inference.NERInference
    elif args.model_type == "TEXTCNN":
        inference_type = classify_inference.ClassifyInference
    else:
        raise TypeError("Unknow model type: {}".format(args.model_type))

    # /Users/tony/Sandbox-Self/MLT/models/11:42:15-2020-06-12
    restore_model_dir = args.restore_model
    inference = inference_type(restore_model_dir, "ckpt")
    # inference = inference_type("/Users/tony/test/pb1_dic_ner", "pb") # pb1_non_dic_ner
    # /Users/tony/test/pb/frozen_mode l.pb

    while (True): 
        print("请输入需要预测的文本：")
        line = sys.stdin.readline().strip()
        # line = "回复支持李开复"
        print("你输入的文本是：{}".format(line))
        if ("q" == line or "exit" == line or "" == line):
            break

        # input数据预处理
        template = features.Template(args.template_file)
        updated_data = query_prepare(line, input2id_, template = template, args = args)
  
        pred = inference.run(updated_data, args.use_dics_flag)
        # pred = neural_tagger.predict(updated_data.feature_list, updated_data.sentence_lengths, model_dir)

        print pred
        if args.model_type == "TEXTCNN":
            print ", ".join(input2id_.id2sentence_label.get(id) for id in pred[0])
        else:
            print ", ".join(input2id_.id2word_label.get(id) for id in pred[0])


def export2pb():
    print("请输入导出pb模型的目标存储路径：")
    output_pb_dir = sys.stdin.readline().strip()
    print("你输入的目标存储路径是：{}".format(output_pb_dir))

    saver = tf.train.import_meta_graph(args.restore_model + ".meta")
    # output_node_names = "line_transform/inference_result,transitions"
    with tf.Session() as sess:
        saver.restore(sess, args.restore_model)
        builder = tf.saved_model.builder.SavedModelBuilder(output_pb_dir)
        builder.add_meta_graph_and_variables(sess, tf.saved_model.tag_constants.SERVING)
        builder.save()

    print("Model已成功导出到：{}".format(output_pb_dir))
    return


def test():
    # ap = args_parser.ArgsParser()
    # ff = ap.parse_args()
    print(args.valid_data)
    print(args.flag_float)
    print(args.flag_int)
    print(args.flag_bool)
    print(args.flag_string)

    with tf.variable_scope("V1",reuse =tf.AUTO_REUSE):
        a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))  
        a2 = tf.Variable(tf.random.normal(shape=[2,3], mean=0, stddev=1), name='a2')  
    with tf.variable_scope('V1', reuse = tf.AUTO_REUSE):  
        a3 = tf.compat.v1.get_variable(name='a1', shape=[1],initializer=tf.constant_initializer(1))  
        a4 = tf.compat.v1.Variable(tf.random.normal(shape=[2,3], mean=0, stddev=1), name='a2')  
    
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())  
        print (a1.name)  
        print (a2.name)   
        print (a3.name)  
        print (a4.name)  

#     with tf.variable_scope("foo", reuse = True):
#     #创建一个常量为1的v
#         v= tf.get_variable('v',[1],initializer = tf.constant_initializer(1.0))
# #因为在foo空间已经创建v的变量，所以下面的代码会报错
# #with tf.variable_scope("foo"）:
# #   v= tf.get_variable('v',[1])
# #在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable的函数将直接获取已声明的变量
# #且调用with tf.variable_scope("foo"）必须是定义的foo空间，而不能是with tf.variable_scope(""）未命名或者其他空间。
#     with tf.variable_scope("foo",reuse =True):
#         v1= tf.get_variable('v',[1])#  不写[1]也可以
#         print(v1==v) #输出为True，代表v1与v是相同的变量
    
#     with tf.Session() as sess:  
#         sess.run(tf.initialize_all_variables())  
#         print (v.name)  
#         print (v1.name)   


def test2():
    a = tf.constant([[1,2],[3,4]], name = "aa")
    b = tf.constant([[2,1],[0,1]], name = "bb")
    c = tf.matmul(a, b, name = "cc")
    d = [1,2]
    e = tf.add(c, d, name = "ee")
    f = tf.reshape(e, [-1, 4], name = "ff")

    with tf.name_scope("line_transform"):
        g = tf.Variable([[2,1],[0,2]], dtype = tf.int32,name = "gg")
        h = tf.add(e, g, name = "hh")

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    flag = False

    with tf.Session() as sess:
        if flag:
            sess.run(init)
            print "original:"
            print f.name
            print h.name

            tensor_ff = tf.get_default_graph().get_tensor_by_name("ff:0")
            print "get tensor"
            print tensor_ff

            tensor_hh = tf.get_default_graph().get_tensor_by_name("line_transform/hh:0")
            print "get tensor"
            print tensor_hh

            # feed_dict = {
            #     g: x[index],
            #     self.Y: y[index],
            #     self.X_len: l[index],
            #     self.keep_prob: keep_prob,
            # }

            res, h_res = sess.run([tensor_hh, h])

            # saver = tf.train.Saver()
            save_path = saver.save(sess, "/Users/tony/test/model.ckpt")

            print res
        else:
            saver = tf.train.import_meta_graph('/Users/tony/test/model.ckpt.meta')
            saver.restore(sess, "/Users/tony/test/model.ckpt")

            graph = tf.get_default_graph()
            tensor = graph.get_tensor_by_name("ff:0")
            print "get tensor from load model"
            print tensor

            h = graph.get_tensor_by_name("line_transform/hh:0")
            h_res = sess.run(h)
            print h

            tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
            for tensor_name in tensor_name_list:
                print(tensor_name)

    # W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
    # b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
 
    # init= tf.initialize_all_variables()
 
    # saver = tf.train.Saver()
 
    # with tf.Session() as sess:
    #     # sess.run(init)
    #     save_path = saver.save(sess, "/Users/tony/test/model.ckpt")
    #     print("Save to path: ", save_path)


# %%
if __name__ == '__main__':
    if args.run_type.upper() == "TRAIN":
        train()
    elif args.run_type.upper() == "INTERACTION":
        interaction()
    elif args.run_type.upper() == "EXPORT2PB":
        export2pb()
    else:
        Logger.warning("The run_type {} is not supported!".format(args.run_type))