# *-* coding:utf-8 *-*
#

import sys, os

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

entity_dict = {"<crf_org>": 2, "<numpure>": 1, "<s_person>": 3, "<person>": 3, "<tm_company>": 2, "<time>": 4, "org": 2,
               "num": 1, "per": 3, "o": 0, "tim": 4}
entity_generalize = {"org": "<crf_org>", "num": "<numpure>", "per": "<s_person>", "tim": "<time>"}


class EventDetector:
    def __init__(self):
        # self.flags=None
        self.classi_dir = None
        self.keywords = None
        self.vocab_processor = None
        self.title_vocab_processor = None
        self.label_dict = None
        self.sess = None
        self.model_operations = None
        self.num_dict = None
        self.default_dict = None
        self.notanyclass_list = None

    def init(self, classi_dir, label_dict_dir, sent_length, checkpoint_dir):
        # self.flags = self.para_define()
        self.classi_dir = classi_dir
        self.keywords = self.get_keywords()
        self.label_dict_dir = label_dict_dir
        self.sent_length = sent_length
        self.checkpoint_dir = checkpoint_dir
        self.vocab_processor, self.title_vocab_processor = self.voc_define()
        self.label_dict = self.label_define()
        self.num_dict = self.read_label("_out")
        self.default_dict = self.read_label("_default")  # “跑路”：1
        self.notanyclass_list = self.read_list("_notanyclass")
        self.sess, self.model_operations = self.load_session()

    def read_label(self, dirstr):
        file = self.label_dict_dir + dirstr
        return data_helpers.load_label_dict(file)

    def read_list(self, dirstr):
        file = self.label_dict_dir + dirstr
        key_filter = set()
        with open(file, "r") as ff:
            for ln in ff.readlines():
                k = ln.strip().decode('utf-8')
                key_filter.add(k)
        return key_filter

    def voc_define(self):
        '''
        加载各种embedding字典
        '''
        vocab_path = os.path.join(self.checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        # vp = learn.preprocessing.VocabularyProcessor(4,tokenizer_fn=tokenizer)
        # vp.restore(vocab_path)

        # title vocabulory
        title_vocab_path = os.path.join(self.checkpoint_dir, "..", "title_vocab")
        title_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(title_vocab_path)
        # title_vp = learn.preprocessing.VocabularyProcessor(4,tokenizer_fn=tokenizer)
        # title_vp.restore(title_vocab_path)
        return vocab_processor, title_vocab_processor

    def label_define(self):
        '''
        标签转换字典
        '''
        label_dict = data_helpers.load_label_dict(self.label_dict_dir)
        # print sorted(label_dict.items(), lambda x, y: cmp(x[1], y[1]))
        return label_dict

    def load_session(self):
        '''
        构建网络结构
        '''
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)  # 通过检查点文件锁定最新的模型
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
        saver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()

        # accuracy = tf.get_collection('accuracy')[0]
        # print accuracy

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        input_y = graph.get_operation_by_name('input_y').outputs[0]
        input_title = graph.get_operation_by_name('input_title').outputs[0]
        input_pos = graph.get_operation_by_name('input_pos').outputs[0]
        input_entity = graph.get_operation_by_name('input_entity').outputs[0]
        input_sf = graph.get_operation_by_name('input_sf').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        model_operations = (
                input_x
                , input_y
                , input_title
                , input_pos
                , input_entity
                , input_sf
                , dropout_keep_prob
                , predictions
                )
        return sess, model_operations

    def load_data_and_labels(self, sentences):
        '''
        读取输入数据
        arg:
            sentences:
                标准结果（不用）
                触发词
                标题分词结果
                触发词起始位置
                触发词结束位置
                内容分词结果
        return:
            input_x: 内容泛化分词结果，[]
            input_title: 标题分词结果，[]
            ny: 标注的分类结果
            beg: 触发词起始位置
            end: 触发词结束位置
            entity: 实体类型
            input_sf: 是否未等
            trigger: 触发词
        '''
        input_x = []
        input_title = []
        input_sf = []
        y = []
        entity = []
        beg = []
        end = []
        triggers = []
        trigger_find = []
        for ln in sentences:
            ln = ln.strip().split('\t')
            #tri = ln[1].decode('utf-8')
            tri = ln[1]
            triggers.append(tri)
            title_list = json.loads(ln[2])
            beg.append(int(ln[3]))
            end.append(int(ln[4]))
            x_list = json.loads(ln[5])
            # 关键词
            trigger = x_list[int(ln[3]): int(ln[4]) + 1]
            # 标题
            # title_list.extend(trigger)
            title_text = [x[1] for x in title_list]
            # print json.dumps(title_text, ensure_ascii=False)
            input_title.append(title_text)
            # 词
            # x_text = [x[1] for x in x_list]
            x_text = [entity_generalize[x[3]] if x[3] in entity_generalize else x[1] for x in x_list]
            # print json.dumps(x_text, ensure_ascii=False)
            input_x.append(x_text)
            # 标签
            label = ln[0].decode()
            y.append(int(label))
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
            sf = 0
            if index and index + 1 < len(x_text):
                if x_text[index + 1] == "是":
                    if "未" in x_text:
                        sf = 1
                    else:
                        sf = 3
                elif x_text[index + 1] == "否":
                    if "未" in x_text:
                        sf = 2
                    else:
                        sf = 4
            input_sf.append(sf)
        ny = np.zeros((len(y), len(self.label_dict)))
        for i, items in enumerate(y, 0):
            ny[i, items] = 1
        return input_x, input_title, ny, beg, end, entity, input_sf, triggers

    def label_reverse(self, y):
        '''
        # label转为类别
        '''
        for k, v in self.label_dict.items():
            if v == y:
                return self.num_dict[k]

    def get_keywords(self):
        keywords = []
        for ln in open(self.classi_dir):
            keyword = ln.strip().decode().split('\t')
            keywords.append(keyword)
        return keywords

    def is_candidate(self, sent,is_eval):
        '''
        在原文中查找关键词
        :param sent: 原文
        :return:
            所有候选触发词
        #规则一：正则匹配(说明|注释|规章|如果|一旦|假定|假设|注：|注:|保证人|担保人)
        '''
        start = len(sent)
        if is_eval == False:
            pattern = re.compile(u'说明|注释|规章|如果|一旦|假定|假设|注：|注:|保证人|担保人|关联方')
            iter = re.finditer(pattern, sent)
            for i in iter:
               # print i
                if start > i.span()[0]:
                    start = i.span()[0]  # 识别的最靠前的位置
        candi_keywords = []
        for w in self.keywords:
           # index = sent.decode().find(w[4])
            index = sent.find(w[4])
            if index != -1:
                if index > start:  # 如果关键词的位置在start后面，则不会成为候选，退出遍历
                    continue
                candi_keywords.append(w[4])

        return candi_keywords

    def corpus_process(self, trigger, segs, result, title,is_eval):
        '''
        语料预处理
        #规则二：黑名单关键词样本去掉
        '''
        #print json.dumps(segs, ensure_ascii=False)
        sample = None
        for i in xrange(len(segs)):
           # print segs[i][1]
            if segs[i][1] in trigger:
                # print ">>>>w", w
                word = ""
                x = 0
                while i + x < len(segs):
                    if segs[i + x][3] != "o":
                        break
                   # word += segs[i + x][1].decode('utf-8','ignore')
                    word += segs[i + x][1]
                   # if wo
                    if word == trigger:
                        if trigger == u'停产' or trigger == u'停工' and (i+x-1)>0:
                            if ssegs[i+x-1][1] == u'半':
                                continue
                        if is_eval == False and word in self.notanyclass_list:  # pass
                            continue
                        # print ">>>>match", word
                        # print("%s\t%s\t%s\t%d\t%d\t%s" % (
                        # result, trigger, json.dumps(title, ensure_ascii=False), i, i + x, json.dumps(segs, ensure_ascii=False)))
                        sample = "%s\t%s\t%s\t%d\t%d\t%s" % (result, trigger, json.dumps(title, ensure_ascii=False), i, i + x, json.dumps(segs, ensure_ascii=False))
                        # print sample
                        break
                    if len(trigger) <= len(word):
                        break
                    x += 1
        return sample

    def get_sample(self,title_seg, body_seg, title, sent,is_eval):
        '''
        判断给定的内容中是否有候选关键词
        arg:
            title_seg: 标题分词结果 [ [A, a, 'n', 'ner'] ]
            body_seg: 内容分词结果，格式同上
            title：标题原文
            sent: 内容原文
            is_eval: 是否只进行统计方法，如果为False则规则方法会生效
        return:
            返回所有候选样本
        '''

        sample_list = []
        result = 0
        tri_list = self.is_candidate(sent,is_eval)
        for tri in tri_list:
            sent_seg =body_seg
            # print json.dumps(sent_seg, ensure_ascii=False)
            sample = self.corpus_process(tri, sent_seg, result, title_seg, is_eval)
            if not sample:
                continue
            sample_list.append(sample)
        return sample_list

    def predeal(self,sample_list):
        x_dev, title_dev, y_dev, dev_beg, dev_end, dev_entity, dev_sf, triggers = \
                self.load_data_and_labels(sample_list)
        x_dev, dev_pos, dev_entity = data_helpers.trigger2mid(\
                x_dev, dev_beg, dev_end, dev_entity, self.sent_length)
        x_dev = np.array(list(self.vocab_processor.transform([" ".join(x) for x in x_dev])))
        title_dev = np.array(list(self.title_vocab_processor.transform([" ".join(x) for x in title_dev])))

        # sess,model_operations = self.load_session()
        sess = self.sess
        model_operations = self.model_operations
        input_x, input_y, input_title, input_pos, input_entity, input_sf, dropout_keep_prob, predictions = model_operations
        feed_dict = {
                input_x: x_dev
                , input_title: title_dev
                , input_pos: dev_pos
                , input_entity: dev_entity
                , input_sf: dev_sf
                , dropout_keep_prob: 1.0
                }
        all_predictions = sess.run(predictions, feed_dict)
        cnt = 0
        for item in triggers:
            if self.default_dict.has_key(item):
                r = self.default_dict[item]
                all_predictions[cnt] = r
            cnt = cnt+1
        return triggers, all_predictions

    def predict(self, title_segs, body_segs, title, body,is_eval = False):
        '''
        预测是否为事件
        arg:
            title_segs: 标题分词结果
            body_segs: 内容分词结果
            title:标题原文
            body: 内容原文
            is_eval: 是否只进行统计方法，如果为False则规则方法会生效
        '''
        sample_list = self.get_sample(title_segs, body_segs, title, body,is_eval)
        '''
        for ll in sample_list:
            print ll
            print "**"
        '''
        triggers, all_predictions = self.predeal(sample_list)
        res={}
        for i in range(len(triggers)):
            clas = self.label_reverse(all_predictions[i])
            if clas != 'notanyclass':
                keys = res.setdefault(clas,[])
                keys.append(triggers[i])
        return res



