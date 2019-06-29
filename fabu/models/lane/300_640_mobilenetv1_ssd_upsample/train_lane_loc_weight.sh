#!/usr/bin/env sh
set -e
    #--weights=./pretrained/train.caffemodel \ 
    #-snapshot=./models/mobile_lane_loc_iter_30000.solverstate \
 ./../caffe-ssd/build/tools/caffe train \
    --weights=./pretrained/train.caffemodel \
    --solver=./net/lane_loc/solver_weight.prototxt --gpu 0,1,2,3 2>&1 | tee log_weight.txt.`date +'%Y-%m-%d_%H-%M-%S'`
