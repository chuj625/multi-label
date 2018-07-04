#!/bin/env python
#coding:utf-8
# Copyright © hanzhonghua@dingfudata.com
#

"""
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


c2num = {}

for ln in open(sys.argv[1]):
    ln = ln.strip()
    ln = ln.decode('utf-8')
    fe = ln.split('\t')
    c = int(fe[0])
    if c not in c2num:
        c2num[c] = 1
    else:
        c2num[c] += 1

print '>>>> {}'.format(sys.argv[1])
for c, num in c2num.items():
    print '      {}:{}'.format(c, num)


