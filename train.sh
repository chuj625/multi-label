#!/bin/sh
#
# Copyright (C) hanzhonghua@dingfudata.com
#

dd=`date +%Y%m%d_%H%M%S`

python bin/train.py > tmp_$dd.log 2>&1 &

