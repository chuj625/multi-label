#*-* coding:utf-8 *-*
#

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import re
import json
import tensorflow as tf
import data_helpers
import csv
from tensorflow.contrib import learn
from sklearn import metrics

#def tokenizer(iterator):
#    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
#    for value in iterator: 
#        yield TOKENIZER_RE.findall(value)

entity_dict = {"<crf_org>":2, "<numpure>":1, "<s_person>":3, "<person>":3, "<tm_company>":2, "<time>":4, "org": 2, "num": 1, "per": 3, "o": 0, "tim": 4}
entity_generalize = {"org": "<crf_org>", "num": "<numpure>", "per": "<s_person>", "tim": "<time>"}

class Predict:
    def __init__(self):
        # self.flags=None
        self.vocab_processor=None
        self.title_vocab_processor = None
        self.label_dict = None
        self.sess = None
        self.model_operations = None

    def init(self, sent_length, checkpoint_dir):
        '''
        '''
        # self.flags = self.para_define()
        self.sent_length = sent_length
        self.checkpoint_dir = checkpoint_dir
        self.vocab_processor = self.voc_define()
        self.label_dict = self.label_define()
        self.sess,self.model_operations = self.load_session()

    def voc_define(self):
        vocab_path = os.path.join(self.checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        #vp = learn.preprocessing.VocabularyProcessor(4,tokenizer_fn=tokenizer)
        #vp.restore(vocab_path)
        
        return vocab_processor

    def label_define(self):
        label_dict = data_helpers.load_label_dict(self.label_dict_dir)
        # print sorted(label_dict.items(), lambda x, y: cmp(x[1], y[1]))
        return label_dict

    def load_session(self):
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()

        # accuracy = tf.get_collection('accuracy')[0]
        # print accuracy

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        input_y = graph.get_operation_by_name('input_y').outputs[0]
        input_entity = graph.get_operation_by_name('input_entity').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        model_operations = \
                (input_x,input_y, input_entity,dropout_keep_prob,predictions)
        return sess,model_operations

    def load_data_and_labels(self,sentences):
        '''
        读取输入数据
        arg:
            sentences: 
                标准结果（不用）
                内容分词结果
                ner结果
        return:
            input_x: 内容泛化分词结果，[]
            entity: 实体类型
            ny: 标注的分类结果
        '''
        input_x = []
        input_title = []
        y = []
        entity = []
        for ln in sentences:
            ln = ln.decode('utf-8')
            fe = ln.strip().split('\t')
            # print json.dumps(title_text, ensure_ascii=False)
            x_text = fe[1].split(' ')
            # print json.dumps(x_text, ensure_ascii=False)
            input_x.append(x_text)
            # 标签
            label = ln[0]
            y.append(int(label))
            # 实体类型
            ent = fe[2].split(' ')
            entity.append(ent)
        ny = np.zeros((len(y), len(self.label_dict)))
        for i, items in enumerate(y, 0):
            ny[i, items] = 1
        return input_x, entity, ny

    def predict(self,sample):
        x_dev, dev_entity,y_dev = self.load_data_and_labels(sample)
        x_dev, dev_entity = \
                data_helpers.data_trim(x_dev, dev_entity, self.sent_length)
        x_dev = np.array(list( \
            self.vocab_processor.transform([" ".join(x) for x in x_dev])))

        sess = self.sess
        input_x, input_y, input_entity, dropout_keep_prob, predictions = \
                self.model_operations
        all_predictions = sess.run(predictions, {input_x: x_dev\
                    , input_entity: dev_entity, dropout_keep_prob: 1.0})

        num = 0
        res = []
        for y in all_predictions:
            res.append(y)
        return res

if __name__ == '__main__':
    pred = Predict()
    pred.init(21, sys.argv[1])


