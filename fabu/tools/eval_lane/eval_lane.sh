root_path=../../models/lane/300_640_mobilenetv1_ssd_upsample
python eval_lane.py --deploy $root_path/deploy_deeper_weight.prototxt --model $root_path/mobile_lane_loc_iter_11000.caffemodel --image ./lane_test.txt --gpu 5 --path ./eval_res/
