#!/bin/sh

mkdir -p snapshot
~/ADAS_caffe/build/tools/caffe train -solver="solver.prototxt" \
-gpu 0,1,2,3,4,5,6  \
2>&1 | tee log.txt.`date +'%Y-%m-%d_%H-%M-%S'`
