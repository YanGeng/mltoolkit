#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict

d = {"a":1, "b":2}
for k, v in d.items():
    print k

dint = defaultdict(int)
dint["a"] += 1
dint["b"] += 1

for k in dint:
    print k
    # print v

d2 = {}
d2["a"] = defaultdict(int)
d2["b"] = defaultdict(int)

d2["a"]["b"] += 1
d2["b"]["a"] += 1
for k, v in d2.items():
    print k
# walk = os.walk("/Users/tony/Sandbox-Self/MLT/outputs/export")
# for root, dirs, files in walk:
#     print files
#     for file_ in files:
#         full_path =  os.path.join(root, file_)


print tf.__version__
# with tf.variable_scope("foo"):
#     #创建一个常量为1的v
#     v= tf.get_variable('v',[1],initializer = tf.constant_initializer(1.0))
# #因为在foo空间已经创建v的变量，所以下面的代码会报错
# #with tf.variable_scope("foo"）:
# #   v= tf.get_variable('v',[1])
# #在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable的函数将直接获取已声明的变量
# #且调用with tf.variable_scope("foo"）必须是定义的foo空间，而不能是with tf.variable_scope(""）未命名或者其他空间。
# with tf.variable_scope("foo",reuse =True):
#     v1= tf.get_variable('v',[1])#  不写[1]也可以
#     print(v1==v) #输出为True，代表v1与v是相同的变量


# p=tf.Variable(tf.random_normal([10,3,2]))#生成10*1的张量
# b = tf.nn.embedding_lookup(p, [1, 9, 3])#查找张量中的序号为1和3的
 
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(b))
#     print "\n"
#     #print(c)
#     print(sess.run(p))
#     print(p)
#     print(type(p))

p = tf.Variable(tf.random_normal([10, 3]))#生成10*1的张量
b = tf.nn.embedding_lookup(p, [1, 3])#查找张量中的序号为1和3的
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    #print(c)
    print(sess.run(p))
    print(p)
    print(type(p))



pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences
test_padding = np.random.randint(5, size=(2, 4, 1))
new_res = pad_sequences(test_padding, maxlen = 7, padding = 'post', value=0.0)



# case 1
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 1*1 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,5,5,3]))
op1 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')


# case 2
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 2*2 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量 
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([2,2,5,1]))
op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

# case 3  
# 输入是1张 3*3 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 1*1 的feature map (不考虑边界)
# 1张图最后输出就是一个 shape为[1,1,1,1] 的张量
input = tf.Variable(tf.random_normal([1,3,3,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
op3 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID') 
 
# case 4
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 3*3 的feature map (不考虑边界)
# 1张图最后输出就是一个 shape为[1,3,3,1] 的张量
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
op4 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  

# case 5  
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是1
# 步长是[1,1,1,1]最后得到一个 5*5 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,5,5,1] 的张量
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,1]))  
op5 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  

# case 6 
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,1,1,1]最后得到一个 5*5 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,5,5,7] 的张量
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')  

# case 7  
# 输入是1张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后得到7个 3*3 的feature map (考虑边界)
# 1张图最后输出就是一个 shape为[1,3,3,7] 的张量
input = tf.Variable(tf.random_normal([1,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')  

# case 8  
# 输入是10 张 5*5 大小的图片，图像通道数是5，卷积核是 3*3 大小，数量是7
# 步长是[1,2,2,1]最后每张图得到7个 3*3 的feature map (考虑边界)
# 10张图最后输出就是一个 shape为[10,3,3,7] 的张量
input = tf.Variable(tf.random_normal([10,5,5,5]))  
filter = tf.Variable(tf.random_normal([3,3,5,7]))  
op8 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')  
  
init = tf.initialize_all_variables() 
with tf.Session() as sess:
    sess.run(init)
    print('*' * 20 + ' op1 ' + '*' * 20)
    print(sess.run(op1))
    print(op1.shape)
    # print('*' * 20 + ' op2 ' + '*' * 20)
    # print(sess.run(op2))
    # print('*' * 20 + ' op3 ' + '*' * 20)
    # print(sess.run(op3))
    # print('*' * 20 + ' op4 ' + '*' * 20)
    # print(sess.run(op4))
    # print('*' * 20 + ' op5 ' + '*' * 20)
    # print(sess.run(op5))
    # print('*' * 20 + ' op6 ' + '*' * 20)
    # print(sess.run(op6))
    # print('*' * 20 + ' op7 ' + '*' * 20)
    # print(sess.run(op7))
    # print('*' * 20 + ' op8 ' + '*' * 20)
    # print(sess.run(op8))