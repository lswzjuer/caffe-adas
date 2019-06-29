#!/usr/bin/env sh
set -e
#  --weights=./pretrained/train.caffemodel
    #-snapshot=./models/ti_iter_27000.solverstate \
 ./../caffe-ssd/build/tools/caffe train \
    --solver=./net/lane_loc/solver_ti.prototxt --gpu 0,1 2>&1 | tee log_ti.txt.`date +'%Y-%m-%d_%H-%M-%S'`
