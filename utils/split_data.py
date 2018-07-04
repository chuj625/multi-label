#!/bin/env python
#coding:utf-8
# Copyright © hanzhonghua@dingfudata.com
#

"""
划分训练集测试集
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from sklearn.utils import shuffle


def split_data(datas):
    '''
    '''
    datas = shuffle(datas, random_state=0)
    train_end = int(0.8*len(datas))
    dev_end = int(0.9*len(datas))
    print "train:{}, dev:{}, test:{}".format(train_end\
            , dev_end-train_end \
            , len(datas)-dev_end)
    return datas[:train_end], datas[train_end:dev_end], datas[dev_end:]

def save(sets, f):
    for s in sets:
        f.write(s.encode('utf-8')+'\n')

if __name__ == '__main__':
    tfile = open(sys.argv[1], 'w')
    dfile = open(sys.argv[2], 'w')
    tstfile = open(sys.argv[3], 'w')

    res = []
    for ln in sys.stdin:
        ln = ln.strip()
        ln = ln.decode('utf-8')
        res.append(ln)
    train_set, dev_set, test_set = split_data(res)
    save(train_set, tfile)
    save(dev_set, dfile)
    save(test_set, tstfile)
    

    

