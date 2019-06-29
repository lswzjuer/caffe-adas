#!/usr/bin/env sh
set -e

./../caffe-ssd/build/tools/caffe train \
    --solver=./net/lane_loc/deeper/solver_weight.prototxt --gpu 6,7 --weights=./pretrained/train.caffemodel 2>&1 | tee log_weight.txt.`date +'%Y-%m-%d_%H-%M-%S'`
 
