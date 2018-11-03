## 模型说明

### caffe层
`caffe_layers` 下存放着模型用到的*shuffle_channel_layer*和*conv_dw_layer*的caffe实现，配置方法同常规的caffe层，具体可参见 [这里](https://github.com/farmingyard/ShuffleNet)。

### 预训练模型

shufflenet_1x_g3.caffemodel为shufflenetV1在ImageNet上预训练的网络。实验证明这个任务如果不适用预训练的网络，则收敛需要非常长的时间

### 模型

size100_l10是指模型输入的图片大小为100x100，网络为shufflenet前10层，size50_l20则是指模型输入图片大小为50x50，用上shufflenet的所有层。实验结果来看，size100_l10的泛化性能会好一些。