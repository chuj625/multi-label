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

def tokenizer(iterator):
    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
    for value in iterator: 
        yield TOKENIZER_RE.findall(value)

entity_dict = {"<crf_org>":2, "<numpure>":1, "<s_person>":3, "<person>":3, "<tm_company>":2, "<time>":4, "org": 2, "num": 1, "per": 3, "o": 0, "tim": 4}
entity_generalize = {"org": "<crf_org>", "num": "<numpure>", "per": "<s_person>", "tim": "<time>"}

class Predict:
    def __init__(self):
        # self.flags=None
        self.vocab_processor=None
        self.title_vocab_processor = None
        self.sess = None
        self.model_operations = None

    def init(self, sent_length, checkpoint_dir):
        '''
        '''
        # self.flags = self.para_define()
        self.sent_length = sent_length
        self.checkpoint_dir = checkpoint_dir
        self.vocab_processor = self.voc_define()
        self.sess,self.model_operations = self.load_session()

    def voc_define(self):
        vocab_path = os.path.join(self.checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        #vp = learn.preprocessing.VocabularyProcessor(4,tokenizer_fn=tokenizer)
        #vp.restore(vocab_path)
        
        return vocab_processor

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

    def predict(self,sample):
        #x_dev, dev_entity,y_dev = data_helpers.load_data_from_list(sample, class_num=10)
        y_dev, x_dev, dev_entity= data_helpers.load_data_from_list(sample, class_num=10)
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
    ins = []
    for ln in sys.stdin:
        ln = ln.strip()
        ins.append(ln)
    res = pred.predict(ins)
    ins = [x.split('\t') for x in ins]
    for r, i in zip(res, ins):
        print '{}\t{}\t{}'.format(i[0], r, '\t'.join(i[1:]))


