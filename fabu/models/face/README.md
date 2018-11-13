## 模型说明

### caffe层

caffe_layers 下存放着模型用到的shuffle_channel_layer和conv_dw_layer的caffe实现，配置方法同常规的caffe层，具体可参见[这里](https://github.com/farmingyard/ShuffleNet)

### 数据

目前数据在/private/zhongxuexian/actionDetect/cropped_data下的train 和 test 文件夹下。

### 脚本

一共有两个脚本用于生成imglist 和 label，分别为filter_train.py 和 generate_label.py， 均存放在/private/zhongxuexian/actionDetect/cropped_data下。

### 预训练模型

shufflenet_1x_g3.caffemodel为shufflenetV1在ImageNet上预训练的网络。实验证明这个任务如果不使用预训练的网络，则收敛需要非常长的时间

### 训练

prototxt 在fabu/models/face/size100_l10_shufflenetv1下, 命令如下：./train.sh

### 模型

size100l10是指模型输入的图片大小为100x100，网络为shufflenet前10层，size50l20则是指模型输入图片大小为50x50，用上shufflenet的所有层。实验结果来看，size100_l10的泛化性能会好一些。
