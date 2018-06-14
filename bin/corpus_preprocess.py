#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

sys.path.append(sys.path[0]+'/../tools')
#print sys.path

import numpy as np
import json
import re
from wordseg_df import Wordseg

class CorpusPreprocess():
    def __init__(self):
        self.keywords = None
        self.wordseg = None

    def init(self, path):
        self.keywords = self.get_keywords(path)
        self.wordseg = Wordseg()

    def get_keywords(self, path):
        keywords = []
        for ln in open(path):
            keyword = ln.strip().decode().split('\t')
            keywords.append(keyword)
        return keywords

    def is_candidate(self, sent):
        '''
        在原文中查找关键词
        :param sent: 原文
        :return:
            所有候选触发词
        '''
        start = len(sent)
        pattern = re.compile('说明|注释|规章|如果|一旦|假定|假设|注：|注:|保证人|担保人')
        iter = re.finditer(pattern,sent)
        for i in iter:
           # print i
            if start>i.span()[0]/3:
                start = i.span()[0]/3#识别的最靠前的位置
            
        candi_keywords = []
        for w in self.keywords:
            index = sent.decode().find(w[4])
            if index!=-1:
                if index > start:#如果关键词的位置在start后面，则不会成为候选，退出遍历
                    break
                candi_keywords.append(w[4])

        return candi_keywords

    def seg(self, sent, granularity=1):
        '''

        :param sample: 需要分词的句子, 粒度
        :return: 分词结果
        '''
        sent_seg = self.wordseg.seg(sent.decode(), granularity)
        return sent_seg

    def corpus_process(self, trigger, segs, result, title):
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
                    word += segs[i + x][1]
                    # print ">>>>word", word
                   # if wo
                    if word == trigger:
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

    def get_sample(self, title, sent):
        '''
        :param sentences: tuple (标题1，句子1）
            tit: 标题
            sent: 句子
        :return:
        '''
        sample_list = []
        result = 0
        tri_list = self.is_candidate(sent)
        for tri in tri_list:
            sent_seg = self.seg(sent)
            # print json.dumps(sent_seg, ensure_ascii=False)
            title_seg = self.seg(title)
            sample = self.corpus_process(tri, sent_seg, result, title_seg)
            if not sample:
                continue
            sample_list.append(sample)
        return sample_list


if __name__ == '__main__':
    # sent = "甲乙丙丁公司已停产,甲乙丙丁公司半停产。"
    # title = "总体情况"
    # result = 0
    cp = CorpusPreprocess()
    cp.init("../data/classi_merge0227")
    # tri_list = cp.is_candidata(sent)
    # sample_list = []
    # for tri in tri_list:
    #     sent_seg = cp.seg(sent)
    #     # print json.dumps(sent_seg, ensure_ascii=False)
    #     title_seg = cp.seg(title)
    #     sample = cp.corpus_process(tri, sent_seg, result, title_seg)
    #     sample_list.append(sample)
    # # print sample_list

    #candidate_samples = ["总体情况\t某公司实际控制人跑路，生产线已停产", "基本信息\t公司实际控制人死亡"]
    sample_list = cp.get_sample("总体情况", "某公司实际控制人跑路，生产线已停产")
    for sample in sample_list:
        print sample
    sample_list = cp.get_sample("基本信息", "公司实际控制人死亡")
    for sample in sample_list:
        print sample
