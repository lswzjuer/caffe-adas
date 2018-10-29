import numpy as np  
import pdb
import sys,os  
import cv2
caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


#net_file= '../../models/od/400_640_mobilenetv1_ssd/deploy.prototxt'  
#caffe_model='../../models/od/400_640_mobilenetv1_ssd/deploy.caffemodel'  

#net_file= '../../models/od/400_640_JDetNet/deploy.prototxt'  
#caffe_model='../../models/od/400_640_JDetNet/ssdJacintoNetV2_iter_130000.caffemodel'  

#net_file = '../../models/lane_od/400_640_ti_conv1a_16/lane_od_deploy.prototxt'
#caffe_model = '../../models/lane_od/400_640_ti_conv1a_16/lane_od.caffemodel'

#net_file = '../../models/lane_od/400_640_ti/lane_od_deploy.prototxt'
#caffe_model = '../../models/lane_od/400_640_ti/lane_od.caffemodel'

net_file = '/private/zhangjiwei/od_lane0930/lane_od_deploy_merge_bn.prototxt'
caffe_model = '/private/zhangjiwei/od_lane0930/lane_od_merge_bn.caffemodel'

test_dir = "input"
out_dir = 'output'

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
          'car', 'bus', 'truck', 'person', 'bicycle', 'motor', 'tricycle', 'block')

def preprocess(src):
    img = cv2.resize(src, (512,384))

    #MobileNetSSD preprocess
    #mean = (104, 117, 123)
    #img = img * 0.007843

    #JDet
    mean = (128, 128, 128)
    img = img - mean

    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]

    box = out['detection_out'][0,0,:,3:7]

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    print box, conf
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    print imgfile
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    print net.blobs['mbox_priorbox'].data.shape
    prior_boxes = net.blobs['mbox_priorbox'].data[0, 0, :]
    print prior_boxes.shape

    '''
    for i in range(0, len(prior_boxes), 4):
        print i / 4, prior_boxes[i], prior_boxes[i+1], prior_boxes[i+2], prior_boxes[i+3] 

    with open('prior_box.h', 'w') as f:
        for i in range(0, len(prior_boxes), 4):
            f.write(str(prior_boxes[i]) + ', ')
            f.write(str(prior_boxes[i+1]) + ', ')
            f.write(str(prior_boxes[i+2]) + ', ')
            f.write(str(prior_boxes[i+3]) + ' \\\n')
    '''

    #print box
    #print cls, conf
    #print net.blobs['conv0'].data.shape
    #print net.blobs['conv0'].data[0][0][0]
    #priors = net.blobs['mbox_priorbox'].data
    #proint priors.shape
    #pdb.set_trace()

    print(len(box))
    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])

       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       #print title
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #cv2.imshow("SSD", origimg)
    cv2.imwrite(os.path.join(out_dir, os.path.split(imgfile)[-1]), origimg)
 
    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
    #break
