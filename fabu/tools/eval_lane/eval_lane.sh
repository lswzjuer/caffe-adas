root_path=/wangwenxiao/ADAS_caffe/fabu/models/lane/400_640_mobilenetv1_ssd_wwx/fixed_finetune
python eval_lane.py --deploy $root_path/test.prototxt --model $root_path/mobilenet-model/_iter_38000.caffemodel --image /wangwenxiao/ADAS_caffe/fabu/data/lane/lane4_val_image_convert_400_640.txt --gpu 1 --path $root_path/res/
