#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

from itertools import combinations
import tensorflow as tf
import numpy as np
from sklearn import metrics

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self \
                , sequence_length\
                , num_classes, vocab_size \
                , embedding_size, embedding_size_entity \
                , filter_sizes, num_filters \
                , sent_length, entity_length \
                , l2_reg_lambda=0.0
            ):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #self.input_title = tf.placeholder(tf.int32, [None, title_length], name="input_title")

        self.input_pos = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos")
        self.input_entity = tf.placeholder(tf.int32, [None, sequence_length], name="input_entity")

        self.input_sf = tf.placeholder(tf.int32, [None,], name="input_sf")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 词向量（预训练）
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]),
                            trainable=False, name="W_word")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = W.assign(self.embedding_placeholder)

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            #self.embedded_title = tf.nn.embedding_lookup(W, self.input_title)

            '''
            # 相对位置embedding[0-(sent_length-1)]
            W = tf.Variable(tf.random_uniform([(sent_length-1)/2+1, embedding_size_pos], -1.0, 1.0),name="W_pos")
            self.embedded_pos = tf.nn.embedding_lookup(W, self.input_pos)
            '''

            # 实体类型embedding
            W = tf.Variable(tf.random_uniform([entity_length, embedding_size_entity], -1.0, 1.0), name="W_entity")
            self.embedded_entity = tf.nn.embedding_lookup(W, self.input_entity)

            # embedding拼接
            self.embedded = tf.concat([self.embedded_chars, self.embedded_entity], 2)

            # embedding矩阵扩展一维
            self.embedded_chars_expanded = tf.expand_dims(self.embedded, -1)
            #self.embedded_title_expanded = tf.expand_dims(self.embedded_title, -1)

            '''
            # 是否embedding
            W = tf.Variable(tf.random_uniform([5, embedding_size_sf], -1.0, 1.0),name="W_sf")
            self.embedded_sf = tf.nn.embedding_lookup(W, self.input_sf)
            '''

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # 词特征
        name_scope = "word"
        embedding_dim = embedding_size + embedding_size_entity
        word_pooled_outputs = self.conv(name_scope, filter_sizes, embedding_dim \
                , num_filters, sequence_length \
                , self.embedded_chars_expanded)
        pooled_outputs.extend(word_pooled_outputs)
        '''
        # 标题特征
        name_scope = "title"
        title_pooled_outputs = self.conv(name_scope, title_filter_sizes, embedding_size, title_num_filters, title_length, self.embedded_title_expanded)
        pooled_outputs.extend(title_pooled_outputs)
        '''
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        '''
        # 合并词(标题)特征和是否特征
        self.h = tf.concat([self.h_pool_flat, self.embedded_sf], 1)
        '''

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # tf.add_to_collection('h_drop', self.h_drop)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # tf.add_to_collection('scores', self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # tf.add_to_collection('predictions', self.predictions)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # tf.add_to_collection('loss', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # tf.add_to_collection('accuracy', self.accuracy)

        '''
        with tf.name_scope("recall"):
            #y_true = tf.argmax(self.input_y, 1)
            self.recall = metrics.recall_score(tf.argmax(self.input_y.eval(), 1), self.predictions, average="micro")

        with tf.name_score("f1_score"):
            #y_true = tf.argmax(self.input_y, 1)
            self.f1_score = metrics.recall_score(tf.argmax(self.input_y.eval(), 1), self.predictions, average="micro")
        '''


    def conv(self, name_scope, filter_sizes, embedding_dim, num_filters
            , sequence_length, embedded):
        '''
        arg:
            name_scope: 名称域
            filter_sizes:
            embedding_dim:
            num_filters:
            sequence_length:
            embedded:
        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("%s_conv-maxpool-%s" % (name_scope, filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  # 截断正态分布随机数
                # W = tf.get_variable(
                #     "%s_conv-maxpool-%s_W" % (name_scope, filter_size),
                #     shape=filter_shape,
                #     initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity, h的shape=[batch, sequence_length - filter_size + 1, 1, num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs, pooled的shape=[batch, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        return pooled_outputs
