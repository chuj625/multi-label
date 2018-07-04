#!/bin/env python
#coding:utf-8
# Copyright © hanzhonghua@dingfudata.com
#

"""
"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def toClass(ss):
    '''
        0 0 0 0 0 0 1 0 0 0 
        转成类别号
    '''
    fs = ss.split(' ')
    res = []
    for i, s in enumerate(fs, start=1):
        if s == '0':
            continue
        if i == 10:
            res.append(0)
            continue
        res.append(i)
    return res

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def conv2normal(ss):
    '''
    去掉分词，转成正常文本
    '''
    ss = ss.replace(' ', '')
    ss = strQ2B(ss)
    return ss


for ln in sys.stdin:
    ln = ln.strip()
    ln = ln.decode('utf-8')
    fe = ln.split('\t')
    sent = conv2normal(fe[0])
    classi = toClass(fe[1])
    for c in classi:
        print '{}\t{}'.format(c, sent)


