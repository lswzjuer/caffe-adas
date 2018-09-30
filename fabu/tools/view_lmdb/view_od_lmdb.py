import os,sys
import numpy as np
import lmdb
import cv2

# caffe_root = '~/code/ssd/python'
# sys.path.append(caffe_root)
import caffe

# lmdb_dir = '/home/yaowanchao/zy_data/lmdb/train_lmdb_640x400_cls7'
# lmdb_dir = "/private/wangwenxiao/adas_dataset/class9_lmdb/adas_od_train_lmdb"
lmdb_dir = "/wangwenxiao/adas_dataset/cityscape/cityscape_person/lmdb/cityscape_person_train_lmdb"
#lmdb_dir = '/home/yaowanchao/zy_data/lmdb/zy146k_zy_train_lmdb_resize'
lmdb_env = lmdb.open(lmdb_dir, readonly=True)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
annot_datum = caffe.proto.caffe_pb2.AnnotatedDatum()

# im_dir = '/nfs/labeler'
im_dir = '/wangwenxiao/adas_dataset/cityscape/fine_crop_image/'

for key, value in lmdb_cursor:
    annot_datum.ParseFromString(value)

    width = annot_datum.datum.width
    height = annot_datum.datum.height
    imfile = key[25:]
    #imfile = key[9:]
    print(width, height, key, imfile.decode())
    im = cv2.imread(os.path.join(str(im_dir), imfile.decode()))
    im = cv2.resize(im, (width, height))

    annot_group = annot_datum.annotation_group
    for annots in annot_group:
        print(annot_group)
        for annot in annots.annotation:
            print(annot)
            b = annot.bbox
            x1 = int(b.xmin * width)
            y1 = int(b.ymin * height)
            x2 = int(b.xmax * width)
            y2 = int(b.ymax * height)
            cv2.rectangle(im, (x1,y1), (x2,y2), (255,0,0))
            #cv2.putText(im, str(annot_group.label), (x1,y1), cv.FONT_HERSHEY_SIMPLEX,2, (0,255,0))
    # cv2.imshow("annot", im)
    cv2.imwrite(imfile.decode()[-30:], im)
    # k = cv2.waitKey(0) & 0xff
    # if k == 27 :
    #    break
