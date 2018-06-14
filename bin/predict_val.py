# *-* coding:utf-8 *-*
#

import sys, os
import json
reload(sys)
sys.setdefaultencoding("utf-8")
import re
import data_helpers
from EventDetector import EventDetector

def tokenizer(iterator):
    TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
    for value in iterator: 
        yield TOKENIZER_RE.findall(value)

pre = EventDetector()
pre.init("../data/classi", "../data/label_dict", 21, "runs/1523266309/checkpoints/")

file_path = "../data/test_data"
label_list = []
res_list = []


def label_define(label_dict_dir):
    label_dict = data_helpers.load_label_dict(label_dict_dir)
    return label_dict

def label_reverse(y,label_dict):
    for k, v in label_dict.items():
        if v == y:
            return k
label_dict = label_define("../data/label_dict")

for ln in open(file_path, 'r'):
    list = ln.strip().split('\t')
    label_list.append(list[0])
    title_segs = json.loads(list[2])
    body_segs = json.loads(list[5])
    title = ""
    body = ""
    for item in title_segs:
        title = title + item[0]
    for item in body_segs:
        body = body + item[0]
    sample_list = pre.get_sample(title_segs, body_segs, title, body, False)
    if sample_list == []:
        print("%s\t%s\t%s\t%s" % ("notanyclass", list[0], title, body))
        continue
    #for sample in sample_list:
    #    print sample
    result_tmp = 0
    this_data = "%s\t%s\t%s\t%d\t%d\t%s" % (result_tmp, list[1], list[2], int(list[3]), int(list[4]), list[5])
    #print this_data
    sample_list_pre = []
    sample_list.append(this_data)
    if sample_list == []:
        triggers, result = pre.predeal(sample_list)
    else:
        result = None
    index = sample_list.index(this_data)
    if index != -1:
    #    print result[index]
        #res = result[index]
        res = label_reverse(result[index],label_dict)
    else:
        res = 'notanyclass'
    res_list.append(res)
    print("%s\t%s\t%s\t%s" % (res, list[0], title, body))


