#!/bin/sh

#-snapshot=snapshot/ssdJacintoNetV2_iter_21000.solverstate \
mkdir -p snapshot
#~/caffe-ssd/build/tools/caffe train -solver="solver.prototxt" \
~/ADAS_caffe/build/tools/caffe train -solver="solver.prototxt" \
-weights=ssdJacintoNetV2_iter_92000.caffemodel \
-gpu 0,1,2,3,4,5,6  \
2>&1 | tee log.txt.`date +'%Y-%m-%d_%H-%M-%S'`
