root_path=/home/sunxiang/code/MobileNetV1-SSD/
python eval_lane.py --deploy $root_path/net/lane_loc/400_640_jsegnet_no_mid_upsample/deploy_ti.prototxt --model $root_path/net/lane_loc/400_640_jsegnet_no_mid_upsample/ti_sim_iter_62000.caffemodel --image ./lane4_val_image_convert_300_640.txt --gpu 2 --path ./../eval_lane/no_mid_upsample/ctx_final/
