#!/bin/sh
#
# Copyright (C) hanzhonghua@dingfudata.com
#

set -ex

#python bin/Predict.py /home/op/hanzhonghua/multi_label/runs/20180702_155316/checkpoints < data/preprocess/test.seg > pred.log
#python bin/eval.py < pred.log

export LD_LIBRARY_PATH=/home/op/app/qa_module/lib:$LD_LIBRARY_PATH

python bin/OriginPredict.py /home/op/hanzhonghua/multi_label/runs/20180702_155316/checkpoints < data/preprocess/test.tmp/sent

