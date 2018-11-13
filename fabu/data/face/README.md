## 司机行为数据说明

### 数据路径
数据存放在`/private/hujinming/data/actionDetect/cropped_data`目录下。其中，该目录下的`public_data` 文件夹存放了从公开数据集中获得的数据，`private_data` 文件夹存放了公司采集的内部数据。而train和test则存放了这些数据的train/test。

### 脚本说明

由于`train`和`test` 文件夹存放了所有的数据，然而原始数据的比例不一定符合我们的需求，可以通过修改运行`filter_train.py` 来对train.txt和test.txt中各个行为数据的组成做一个采样。采样后，可运行`generate_label.py`生成train.txt和test.txt中需要的图片的label。

