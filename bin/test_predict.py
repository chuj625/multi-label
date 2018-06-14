#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import re
import numpy as np
import json
from itertools import izip

from EventDetector import EventDetector

def tokenizer(iterator):
    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
    for value in iterator: 
        yield TOKENIZER_RE.findall(value)
pre = EventDetector()
pre.init("../data/classi", "../data/label_dict", 21, "runs/1523266309/checkpoints/")
title_segs = u'[["三", "3.000000", "m", "num"], ["、", "、", "fu", "o"], ["问题", "问题", "n", "o"], ["发生", "发生", "v", "o"], ["的", "的", "u", "o"], ["主要", "主要", "a", "o"], ["原因", "原因", "n", "o"], ["分析", "分析", "v", "o"], ["1", "1.000000", "m", "num"], ["、", "、", "fu", "o"], ["内部", "内部", "n", "o"], ["原因", "原因", "n", "o"]]'
body_segs = u'[["借款", "借款", "v", "o"], ["人", "人", "n", "o"], ["经营", "经营", "v", "o"], ["管理", "管理", "v", "o"], ["不善", "不善", "a", "o"], [",", ",", "fu", "o"], ["资金", "资金", "n", "o"], ["投资", "投资", "v", "o"], ["于", "于", "p", "o"], ["房", "房", "n", "o"], ["地产", "地产", "n", "o"], ["开发", "开发", "v", "o"], ["失败", "失败", "v", "o"], [",", ",", "fu", "o"], ["资金", "资金", "n", "o"], ["链", "链", "n", "o"], ["断裂", "断裂", "v", "o"], ["。", "。", "fu", "o"], ["实际", "实际", "a", "o"], ["控制", "控制", "v", "o"], ["人", "人", "n", "o"], ["蔡", "蔡", "nr", "o"], ["三", "3.000000", "m", "num"], ["携", "携", "v", "o"], ["妻", "妻", "n", "o"], ["、", "、", "fu", "o"], ["子", "子", "n", "o"], ["“", "“", "fu", "o"], ["跑路", "跑路", "v", "o"], ["”", "”", "fu", "o"], ["至今", "至今", "ad", "o"], ["仍", "仍", "ad", "o"], ["下落", "下落", "v", "o"], ["不明", "不明", "a", "o"], [",", ",", "fu", "o"], ["借款", "借款", "v", "o"], ["人", "人", "n", "o"], ["及", "及", "c", "o"], ["关联", "关联", "v", "o"], ["企业", "企业", "n", "o"], ["(", "(", "fu", "o"], ["保证", "保证", "v", "o"], ["人", "人", "n", "o"], [")", ")", "fu", "o"], ["关停", "关停", "v", "o"], ["。", "。", "fu", "o"]]'

title=""
body=""

title_segs = json.loads(title_segs)
body_segs = json.loads(body_segs)
for item in title_segs:
    title = title + item[0]
for item in body_segs:
    body = body + item[0]
print "=========================="
print title
print body
res = pre.predict(title_segs,body_segs,title.decode('utf-8'),body.decode('utf-8'))
print 'output: {}'.format(json.dumps(res, ensure_ascii=False))
