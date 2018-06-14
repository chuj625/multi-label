#*-* coding:utf-8 *-*

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import re
import numpy as np
import json

from itertools import izip

from Predict import Predict
from corpus_preprocess import CorpusPreprocess

def tokenizer(iterator):
    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
    for value in iterator: 
        yield TOKENIZER_RE.findall(value)

# 预处理
cp = CorpusPreprocess()
cp.init("../data/classi_merge0227")
candidate_samples = ["总体情况\t某公司实际控制人跑路，生产线已停产", "基本信息\t公司实际控制人死亡"]
for ln in open('tt.txt'):
    ln = ln.strip()
    fe = ln.split('\t')
    if len(fe) <2:
        continue
    candidate_samples = fe
    sample_list = cp.get_sample(candidate_samples[0], candidate_samples[1])
    for sample in sample_list:
        print sample

