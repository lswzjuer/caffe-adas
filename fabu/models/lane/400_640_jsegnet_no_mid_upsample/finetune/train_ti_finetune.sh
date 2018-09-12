#!/usr/bin/env sh
set -e
    #-weights=ti_sim_iter_299000.caffemodel
    #-snapshot=./models/ti_sim_iter_60000.solverstate \
 ~/caffe-ssd/build/tools/caffe train \
    -weights=../../../od/400_640_JDetNet/ssdJacintoNetV2_iter_130000.caffemodel \
    --solver=solver_ti_finetune.prototxt --gpu 1,2,3,4,5,6,7 2>&1 | tee log_ti.txt.`date +'%Y-%m-%d_%H-%M-%S'`
