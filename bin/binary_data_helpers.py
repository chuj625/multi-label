#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import re
import json

# label_dict = {'notanyclass':0}
entity_dict = {"org": 2, "num": 1, "per": 3, "o": 0, "tim": 4}

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_label_id(label, label_dict, is_train):
    """
    获得label下标。
    Args:
        label:  标签

    Returns:
        下标, int
    """
    if is_train:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
    return label_dict[label]

def get_entity_id(entity, is_train):
    """
    获得label下标。
    Args:
        label:  标签

    Returns:
        下标, int
    """
    if is_train:
        if entity not in entity_dict:
            entity_dict[entity] = len(entity_dict)
    return entity_dict[entity]

def load_data_and_labels(file):
    input_x = []
    input_title = []
    input_sf = []
    y = []
    entity = []
    beg = []
    end = []
    vocab = set()
    for ln in open(file, 'r'):
        ln = ln.strip().split('\t')
        title_list = json.loads(ln[2])
        beg.append(int(ln[3]))
        end.append(int(ln[4]))
        x_list = json.loads(ln[5])
        # 关键词
        trigger = x_list[int(ln[3]) : int(ln[4])+1]
        # 标题
        # title_list.extend(trigger)
        title_text = [x[1] for x in title_list]
        # print json.dumps(title_text, ensure_ascii=False)
        input_title.append(title_text)
        # 词
        x_text = [x[1] for x in x_list]
        # print json.dumps(x_text, ensure_ascii=False)
        vocab = vocab|set(x_text)
        input_x.append(x_text)
        # 标签
        label = int(ln[0])
        y.append(label)
        # label = ln[0].decode()
        # y.append(get_label_id(label, label_dict, is_train))
        # 实体类型
        ent = [entity_dict[x[3]] for x in x_list]
        entity.append(ent)
        # 是、否
        # print json.dumps(x_text, ensure_ascii=False)
        index = None
        if "■" in x_text:
            index = x_text.index("■")
        elif " " in x_text:
            index = x_text.index(" ")
        # print index
        if index and index + 1 < len(x_text):
            if x_text[index+1] == "是":
                if "未" in x_text:
                    sf = 1
                else:
                    sf = 3
            elif x_text[index+1] == "否":
                if "未" in x_text:
                    sf = 2
                else:
                    sf = 4
        else:
            sf = 0
        input_sf.append(sf)
    # print sorted(label_dict.items(), lambda x, y: cmp(x[1], y[1]))
    # print json.dumps(entity_dict, ensure_ascii=False)
    # print entity
    ny = np.zeros((len(y), 2))
    for i, items in enumerate(y, 0):
        ny[i, items] = 1
    # with tf.Session() as sess:
    #     ny = sess.run(tf.one_hot(y, len(label_dict), 1, 0))
    # return vocab, input_x, input_title, ny, beg, end, entity
    return input_x, input_title, ny, beg, end, entity, input_sf

def load_label_dict(file):
    f = open(file, "r")
    js = json.loads(f.readline())
    return js

def save_label_dict(file, label_dict):
    f = open(file, "w")
    f.write("%s" % json.dumps(label_dict, ensure_ascii=False))
    f.close()

# Data Preparation
def trigger2mid (input_x, beg, end, input_entity, n):
    x_list = []
    pos_list = []
    entity_list = []
    for i in xrange(len(input_x)):
        x = ["<UNK>"] * ((n-1)/2) + input_x[i] + ["<UNK>"] * ((n-1)/2)
        entity = [0] * ((n-1)/2) + input_entity[i] + [0] * ((n-1)/2)
        beg[i] += ((n-1)/2)
        end[i] += ((n-1)/2)
        # 得到相对位置
        pos = []
        for j in xrange(len(x)):
            if j < beg[i]:
                pos.append(beg[i]-j)
            elif j > end[i]:
                pos.append(j-end[i])
            else:
                pos.append(0)
        # 新的关键词下标
        beg_new = (n-1)/2-(end[i]-beg[i]+1)/2
        # 子列表的下标
        index_beg = beg[i]-beg_new
        index_end = index_beg + (n-1)
        # 获取子列表
        x = x[index_beg:index_end+1]
        # print json.dumps(x, ensure_ascii=False)
        x_list.append(x)
        pos = pos[index_beg:index_end+1]
        # print pos
        pos_list.append(pos)
        entity = entity[index_beg:index_end+1]
        # print entity
        entity_list.append(entity)
    return x_list, pos_list, entity_list

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    load_data_and_labels("../data/test_filter_data")