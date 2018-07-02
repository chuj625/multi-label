#!/bin/sh
#
# Copyright (C) hanzhonghua@dingfudata.com
#


preprocess=data/preprocess

rm -rf ${preprocess}
mkdir -p $preprocess

cat data/mltc.all.seg.test.txt | iconv -f gbk -t utf-8 > $preprocess/origin
cat $preprocess/origin | cut -f1 > $preprocess/sents
cat $preprocess/origin | cut -f3 > $preprocess/labels

cat $preprocess/origin | python utils/split_data.py $preprocess/train $preprocess/dev $preprocess/test

# 转成多分类问题
cat $preprocess/train | python  utils/c2multi_classi.py > $preprocess/train.std
python utils/class_status.py $preprocess/train.std
cat $preprocess/dev | python utils/c2multi_classi.py > $preprocess/dev.std
python utils/class_status.py $preprocess/dev.std
cat $preprocess/test | python utils/c2multi_classi.py > $preprocess/test.std
python utils/class_status.py $preprocess/test.std

#分词
wordseg=/home/hanzhonghua/project/multi_label/tools/wordseg_df/wordseg.sh
seg()
{
    ins=$1
    tmp=$2
    out=$3
    echo wordseg $ins
    rm -rf $tmp
    mkdir -p $tmp
    cut -f1 $ins > $tmp/std
    cut -f2 $ins > $tmp/sent
    # 大粒度分词
    cat $tmp/sent | $wordseg 1 > $tmp/sent.seg
    paste $tmp/std $tmp/sent.seg | python utils/wordseg2plain.py > $out
}
seg $preprocess/train.std $preprocess/train.tmp $preprocess/train.seg
seg $preprocess/dev.std $preprocess/dev.tmp $preprocess/dev.seg
seg $preprocess/test.std $preprocess/test.tmp $preprocess/test.seg


