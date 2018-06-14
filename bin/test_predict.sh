#!/bin/sh
#
# Copyright (C) hanzhonghua@dingfudata.com
#

export LD_LIBRARY_PATH=/home/op/app/qa_module/lib

source ../venv/bin/activate

python  test_predict.py

