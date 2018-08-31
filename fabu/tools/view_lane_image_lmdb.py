import os,sys
import numpy as np  
import lmdb
import cv2

caffe_root = '~/code/ssd/python'
sys.path.append(caffe_root)
import caffe  

lmdb_dir = '/private/sunxiang/data/LANE/LMDB/300_640/lane4_val_image_lmdb_300_640'
lmdb_env = lmdb.open(lmdb_dir, readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

cnt = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum).squeeze()
    print cnt, data.shape

    cv2.imwrite('image' + str(cnt) + '.png', data[0])

    cnt += 1
