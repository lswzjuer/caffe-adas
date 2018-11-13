~/caffe/build/tools/caffe train \
  --model shufflenet_v1_l10_train_val.prototxt \
  --weights ../shufflenet_1x_g3.caffemodel \
  --solver solver_shuffle_v1_l10.prototxt \
  --gpu=0,1,2,3 \
  2>&1 | tee log.txt.`date +'%Y-%m-%d_%H-%M-%S'`
