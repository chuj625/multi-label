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

#f = open("../data/runs/1518073041/prediction", "r")
f = open(sys.argv[1], "r")
num_total = 0
num_r = 0
predict = []
gold = []
for ln in f:
    ln = ln.strip().decode().split("\t")
    pre = int(ln[0])
    gol = int(ln[1])
    num_total += 1
    if pre == gol:
        num_r += 1
    predict.append(pre)
    gold.append(gol)
# print num_r
# print num_total
print float(num_r)/float(num_total)
print(metrics.classification_report(gold, predict))
