#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import re
import json

# label_dict = {'notanyclass':0}
entity_dict = {"<crf_org>": 2, "<numpure>": 1, "<person>": 3, "o": 0, "<time>": 4}
entity_generalize = {"org": "<crf_org>", "num": "<numpure>", "per": "<s_person>", "tim": "<time>"}

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

def load_data_and_labels(file, class_num, is_train=False):
    '''
    读取输入数据
    arg:
        file: 输入数据
        class_num: 类别数量
        is_train: 是否为训练过程，如果为训练过程可能增加新标签
    return:
        ny: 标注的分类结果
        input_x: 内容分词结果，[]
        entity: 实体类型
    '''
    input_x = []
    y = []
    entity = []
    vocab = set()
    triggers = []
    for ln in open(file, 'r'):
        ln = ln.decode('utf-8')
        fe = ln.strip().split('\t')
        if len(fe) <3:
            continue
        w = fe[1].split(' ')
        e = fe[2].split(' ')
        yy = int(fe[0])
        # 词
        #x_text = [entity_generalize[x[3]] if x[3] in entity_generalize else x[1] for x in x_list]
        # print json.dumps(x_text, ensure_ascii=False)
        vocab = vocab|set(w)
        input_x.append(w)   # 不泛化
        # 标签
        y.append(yy)
        # 实体类型
        e = [get_entity_id(x, is_train) for x in e]
        entity.append(e)
    ny = np.zeros((len(y), class_num))
    for i, items in enumerate(y, 0):
        ny[i, items] = 1
    return ny, input_x, entity

def load_label_dict(file):
    f = open(file, "r")
    js = json.loads(f.readline().strip())
    return js

def save_label_dict(file, label_dict):
    f = open(file, "w")
    f.write("%s" % json.dumps(label_dict, ensure_ascii=False))
    f.close()

# Data Preparation
def trigger2mid (input_x, beg, end, input_entity, n):
    '''
    将处发词放在文本中间, 提取窗口
    '''
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

def data_trim(input_x, input_entity, n):
    '''
    将数据拉齐，统一截取长度为n
    arg:
        input_x: 词
        input_entity: ner
        n: 最大长度
    return
        xres: 词
        eres: ner
    '''
    xres = []
    eres = []
    for x, e in zip(input_x, input_entity):
        clen = len(x)
        if n <= clen:
            xres.append(x[:n])
            eres.append(e[:n])
        else:
            xres.append(x + ["<UNK>"]*(n-clen))
            eres.append(e + [0] * (n-clen))
    return xres, eres


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    data: 输入数据
    batch_size: 每个batch大小
    num_epochs: epoch数量
    shuffle: 是否进行shuffle, 默认进行shffle
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print "epoch: {}#".format(epoch)
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
    ny, nx, ne = load_data_and_labels("data/preprocess/dev.seg", class_num=10)
    for y, x, e in zip(ny, nx, ne):
        print y, json.dumps(x, ensure_ascii=False), e
