# *-* coding:utf-8 *-*
#

import sys, os
import json
reload(sys)
sys.setdefaultencoding("utf-8")

import json
import numpy as np
from sklearn import metrics
from collections import Counter

def eval_list(res, not_count_set=None):
    '''
    '''
    num_total = 0
    num_r = 0
    num_fenzi = 0
    num_pre = 0
    num_gol = 0
    predict = []
    gold = []
    for r in res:
        fe = r.split('\t')
        pre = fe[0]
        gol = fe[1]
        predict.append(pre)
        gold.append(gol)

    for g, p in zip(gold, predict):
        pre = p
        gol = g
        if pre == gol and (not not_count_set or pre not in not_count_set):
            num_fenzi += 1
        if not not_count_set or gol not in not_count_set:
            num_gol += 1
        if not not_count_set or pre not in not_count_set:
            num_pre += 1
    precision_micro = float(num_fenzi) / float(num_pre)        
    recall_micro = float(num_fenzi) / float(num_gol)
    f1 = 2.0 / (1.0/precision_micro + 1.0/recall_micro)
    print "微平均准确率", precision_micro
    print "微平均召回率",recall_micro
    print "f1值:", f1
    print("*"*66)
    report = metrics.classification_report(gold, predict)
    print(report.encode("utf-8"))
    print(len(gold))

if __name__ == "__main__":
    res = []
    for ln in sys.stdin:
        ln = ln.strip()
        ln = ln.decode('utf-8')
        res.append(ln)
    eval_list(res)
