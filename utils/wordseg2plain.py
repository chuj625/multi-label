#!/bin/env python
#coding:utf-8
# Copyright © hanzhonghua@dingfudata.com
#

"""
把分词结果转成平面的，方便看
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


import json

for ln in sys.stdin:
    w = []
    ner = []
    ln = ln.strip()
    ln = ln.decode('utf-8')
    fe = ln.split('\t')
    seg = json.loads(fe[1])
    for s in seg:
        w.append(s[0])
        ner.append(s[3])
    print '{}\t{}\t{}'.format(fe[0]
            , ' '.join(w), ' '.join(ner))





