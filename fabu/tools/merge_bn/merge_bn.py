#coding:utf-8
import numpy as np  
import sys,os 

caffe_root = '/usr/local/caffe-ssd/python'
sys.path.insert(0, caffe_root)  
import caffe  

caffe.set_device(0)
caffe.set_mode_gpu()

def merge_bn(net, nob):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    for key in net.params.keys():
        if type(net.params[key]) is caffe._caffe.BlobVec:
            if key.endswith("/bn") or key.endswith("/scale"):
                continue
            else:
                conv = net.params[key]
                if not key + "/bn" in net.params.keys():
                    for i, w in enumerate(conv):
                        nob.params[key][i].data[...] = w.data
                else:
                    bn = net.params[key + "/bn"]
                    scale = net.params[key + "/scale"]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    nob.params[key][0].data[...] = wt
                    nob.params[key][1].data[...] = bias

def inference(net1,net2,test_image):
    mu = np.array([128, 128, 128])
    transformer = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
    #  h  w c -> c h w
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_mean('data', mu)
    #self.transformer.set_input_scale('data', 0.007843)
    # rgb -> bgr 
    transformer.set_channel_swap('data', (2, 1, 0))
    try:
        img = caffe.io.load_image(test_image)     
    except IOError:
        print("Open failed")
    else:
        transformed_image = transformer.preprocess('data', img)
        net1.blobs['data'].data[...] = transformed_image
        net2.blobs['data'].data[...] = transformed_image
        # inference
        net1.forward()
        net2.forward()

def compare_data(net1,net2,test_image):
    '''
    net1: the origin net 
    net2: the net without bn model
    '''
    def check_layer_data(data1,data2):
        res=data1==data2
        if res.all():
            return True
        else:
            return False
    # 所有含有参数的层的名字
    net1keys=net1.params.keys()
    net2keys=net2.params.keys()
    conv_layers=[]
    for key in net1keys:
        if type(net1.params[key]) is caffe._caffe.BlobVec:
            # 保存卷积层的层名
            if (not (key.endswith("/bn") or key.endswith("/scale"))) and (key in net2keys):
                conv_layers.append(key)

    # load image and inference
    inference(net1,net2,test_image)

    # 直接对比原网络和去掉BN,SCALE之后的网络每个blob的数据是否相等
    # caffe里面 conv+bn+scale+relu都是在一个blob之中，因此直接用conv名字取blob数据
    for layer in conv_layers:
        data1=net1.blobs[layer].data[0]
        data2=net1.blobs[layer].data[0]
        print(data1.shape,data2.shape)
        if check_layer_data(data1,data2):
            print("LAYER: {} is euqal".format(layer))
        else:
            print("!!!!!!!!  LAYER: {} is not euqal !!!!!!!!!! ".format(layer))

if __name__ == '__main__':
    # laneModel = '../../ssd_lane_99000/train_iter_120000.caffemodel'
    # lanePro = '../../ssd_lane_99000/ssd_lane_deploy.prototxt'

    # deploy_proto ='../../ssd_lane_99000/ssd_lane_deploy_nobn.prototxt'
    # save_model = '../../ssd_lane_99000/ssd_lane_deploy_nobn.caffemodel'


    laneodModel='../../ssd_lane_99000/deploy_all_fuse.caffemodel'
    laneodproto='../../ssd_lane_99000/deploy_all.prototxt'

    deploy_all_proto='../../ssd_lane_99000/deploy_all_nobn.prototxt'
    save_model = '../../ssd_lane_99000/deploy_all_nobn.caffemodel'

    net = caffe.Net(laneodproto, laneodModel, caffe.TEST)  
    net_deploy = caffe.Net(deploy_all_proto, caffe.TEST)

    # merge bn and save the new model
    merge_bn(net, net_deploy)
    net_deploy.save(save_model)

    # Verify the correctness of the new model
    test_image="../../image/adas_image/201711241049_00002180_1511491752356.jpg"
    compare_data(net,net_deploy,test_image)

