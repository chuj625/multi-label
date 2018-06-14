#*-* coding:utf-8 *-*
#

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json
import numpy as np
from sklearn import metrics

np.set_printoptions(threshold='nan')

def reverse(label, label_dict):
    for k, v in label_dict.items():
        if k == label:
            return int(v)

label_dict = json.loads(open("../data/label_dict", "r").readline())
# print json.dumps(label_dict)
#f = open("../data/runs/1518073041/prediction", "r")
f = open(sys.argv[1], "r")
num_total = 0
num_r = 0
predict = []
gold = []
for ln in f:
    ln = ln.strip().decode().split("\t")
    pre = ln[0]
    gol = ln[1]
    num_total += 1
    if pre == gol:
        num_r += 1
    predict.append(reverse(pre, label_dict))
    gold.append(reverse(gol, label_dict))
# print num_r
# print num_total
print float(num_r)/float(num_total)
print(metrics.classification_report(gold, predict))
