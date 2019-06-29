#!/usr/bin/env sh
set -e
    #-weights=../../../od/400_640_JDetNet/ssdJacintoNetV2_iter_92000.caffemodel \
    #-weights=ti_sim_finetune_iter_40000.caffemodel \
    #-snapshot=./snapshot/ti_sim_finetune_iter_40000.solverstate \
 #~/caffe-ssd/build/tools/caffe train \
 ~/ADAS_caffe/build/tools/caffe train \
    -snapshot=./snapshot/ti_sim_finetune_iter_13500.solverstate \
    --solver=solver_ti_finetune.prototxt --gpu 0,1,2,3,4,5,6,7 2>&1 | tee log_ti.txt.`date +'%Y-%m-%d_%H-%M-%S'`
