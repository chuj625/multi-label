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


sys.path.append('/home/hanzhonghua/project/multi_label/tools')
from Predict import Predict
from wordseg_df import Wordseg

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
        ln = ln.decode('utf-8')
        res = op.predict(ln)
        print '{}\t{}'.format(res, ln)

