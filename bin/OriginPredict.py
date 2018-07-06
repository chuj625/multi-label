#!/bin/env python
#coding:utf-8
# Copyright © hanzhonghua@dingfudata.com
#

"""
将分词集成进来，完整的分类
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os

cur_dir= os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.sep.join([cur_dir, '../tools']))
from Predict import Predict
from wordseg_df import Wordseg

import re

def tokenizer(iterator):
    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
    for value in iterator: 
        yield TOKENIZER_RE.findall(value)

class OriginPredict:
    def __init__(self):
        self.model = None
    
    def init(self, checkpoint_dir):
        self.model = Predict()
        self.model.init(21, checkpoint_dir)
        self.wordseg = Wordseg()

    def predict(self, ins):
        sent_seg = self.wordseg.seg(ins, 1)
        segs = [x[0] for x in sent_seg]
        ners = [x[3] for x in sent_seg]
        ni = '0\t{}\t{}'.format(' '.join(segs), ' '.join(ners))
        res = self.model.predict([ni])
        return res[0]
        
if __name__ == '__main__':
    op = OriginPredict()
    op.init(sys.argv[1])
    for ln in sys.stdin:
        ln = ln.strip()
        if not ln:
            continue
        ln = ln.decode('utf-8')
        res = op.predict(ln)
        print '{}\t{}'.format(res, ln)

