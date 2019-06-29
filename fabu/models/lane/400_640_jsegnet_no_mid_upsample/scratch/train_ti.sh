#!/usr/bin/env sh
set -e
#  --weights=./pretrained/train.caffemodel
    #-snapshot=./models/ti_iter_27000.solverstate \
 ~/ADAS_caffe/build/tools/caffe train \
    --solver=solver_ti.prototxt --gpu 0,1,2,3,4,5,6,7 2>&1 | tee log_ti.txt.`date +'%Y-%m-%d_%H-%M-%S'`
