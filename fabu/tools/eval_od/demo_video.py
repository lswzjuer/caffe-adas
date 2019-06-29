import numpy as np  
import pdb
import sys,os  
import cv2
caffe_root = '/home/yaochuanqi/work/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


#net_file= '../../models/od/400_640_mobilenetv1_ssd/deploy.prototxt'  
#caffe_model='../../models/od/400_640_mobilenetv1_ssd/deploy.caffemodel'  
net_file = '../../models/lane_od/400_640_ti_conv1a_16/lane_od_deploy.prototxt'
caffe_model = '../../models/lane_od/400_640_ti_conv1a_16/lane_od.caffemodel'

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
    #img = cv2.resize(src, (640,400))
    img = src[96:, 64: 576]
    print img.shape

    img = img - 128.0
    #mean = (104, 117, 123)
    #img = img - mean

    #img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(origimg, writer):
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)

    #print box
    #print cls, conf
    #print net.blobs['conv0'].data.shape
    #print net.blobs['conv0'].data[0][0][0]
    #priors = net.blobs['mbox_priorbox'].data
    #print priors.shape
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
    writer.write(origimg)
 
    return True

cap = cv2.VideoCapture('20180818.avi')
out = cv2.VideoWriter('20180818_res.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (640,480))

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        detect(frame, out)
    else: 
        break

# When everything done, release the video capture object
cap.release()
out.release()

