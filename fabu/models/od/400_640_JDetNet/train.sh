#!/bin/sh

mkdir -p snapshot
~/ADAS_caffe/build/tools/caffe train -solver="solver.prototxt" \
-gpu 2  \
2>&1 | tee log.txt.`date +'%Y-%m-%d_%H-%M-%S'`
