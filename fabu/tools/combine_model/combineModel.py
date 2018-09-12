
# # Combine two pretrained net into one

import caffe
import numpy as np
import sys, os

caffe.set_device(4)
caffe.set_mode_gpu()

def init():
    laneModel = '/home/yaowanchao/ADAS_caffe/fabu/models/lane/400_640_jsegnet_no_mid_upsample/finetune/ti_sim_finetune_iter_28000.caffemodel'
    lanePro = '/home/yaowanchao/ADAS_caffe/fabu/models/lane/400_640_jsegnet_no_mid_upsample/deploy_ti.prototxt'

    odModel = '/home/yaowanchao/ADAS_caffe/fabu/models/od/400_640_JDetNet/ssdJacintoNetV2_iter_130000.caffemodel'
    odPro = '/home/yaowanchao/ADAS_caffe/fabu/models/od/400_640_JDetNet/deploy.prototxt'

    global lane_od_model
    lane_od_model = './lane_od.caffemodel'   # the combinated model that will be generated
    lane_od_pro = './lane_od_deploy.prototxt' # the combinated net architecture, need to be predefied

    global laneNet, odNet, lane_od_net
    laneNet = caffe.Net(lanePro, laneModel, caffe.TEST)
    odNet = caffe.Net(odPro, odModel, caffe.TEST)
    lane_od_net = caffe.Net(lane_od_pro, caffe.TEST)


def printBlobs(inputNet):
    """Print blob name and shape in inputNet
    Params:
        inputNet: net 
    """
    print '>>>> print blobs'
    for layer_name, blob in inputNet.blobs.iteritems():
        print layer_name + '    ' + str(blob.data.shape)

def printParams(inputNet):
    """Print layer name and params in inputNet
    Params:
        inputNet: net 
    """    
    print '>>>> print parameters'
    for layer_name, param in inputNet.params.iteritems():
        print layer_name + '    ' + str(param[0].data.shape) #, str(param[1].data.shape)

def checkSuffix(layerName, suffix):
    """check whether suffix of layerName is the specific string
    """

    ## BN layer in caffe has 3 params, mean value, variance value, moving cofficient
    if layerName.endswith(suffix) or layerName.endswith(suffix):
        return True
    return False

def paramsNum(srcNet, layerName):
    """Get parameters number in the specific layer in srcNet
    params:
        srcNet: input net
        layerName: layer name
    return:
        num: the parameters number
    """
    return len(srcNet.params[layerName])

def printData(srcNet, layerName):
    """print the weight or bias of specific layer in the input net
    """
    n_params = paramsNum(srcNet, layerName)
    for i in range(0, n_params):
        print srcNet.params[layerName][i].data[...]

def getLayers(inputNet):
    """decode all layers of the inputNet
    params:
        inputNet: net needed to be parsed
    return:
        inputNet_layers: all layer names in the inputNet
    """
    inputNet_layers = []
    for key in inputNet.params.iterkeys():
        if type(inputNet.params[key]) is caffe._caffe.BlobVec:
            #print key
            inputNet_layers.append(key)
    return inputNet_layers


def writeData(inputNet, resFile):
    """Write the weight or bias of specific layer to file
    params:
        inputNet: input net
        resFile: file path
    """
    inputLayers = getLayers(inputNet)
    res_fid = open(resFile, "w")
    for layerName, params in inputNet.params.iteritems():
        n_params = paramsNum(inputNet, layerName)
        for i in range(0, n_params):
            value = params[i].data[...]
            print >> res_fid, layerName, "\nvalue: ", value

def copyData(dstNet, srcNet, layerName):
    """Copy data in specific layer from srcNet to dstNet
    params:
        srcNet: source net
        dstNet: destination net
        layerName: layer name need to be copied
    """
    n_params = len(srcNet.params[layerName])
    #print "n_params: ", n_params
    for i in range(0, n_params):
        dstNet.params[layerName][i].data[...] = srcNet.params[layerName][i].data[...]
        #print "layerName: ", layerName, W.shape, b.shape
        
def checkData(inNet1, inNet2, layerName):
    """check whether the data in the same layer of different Net is equal or not.
    params: 
        inNet1: input net 1
        inNet2: input net 2
        layerName: layer name needed to be compared
        
    """
    n_params = paramsNum(inNet1, layerName)
    for i in range(0, n_params):
        res = (inNet1.params[layerName][i].data[...] == inNet2.params[layerName][i].data[...])
        if res.all():
            print "param %d in %s-%s is equal." % (i, inNet1, inNet2)
        else:
            print "param %d in %s-%s is not equal." % (i, inNet1, inNet2)


def combinateModel(dstNet, srcNet1, srcNet2, savemodel):
    """Combinate srcModel1 and srcModel2 into dstModel
    Params:
        dstNet: output combinated model prototxt
        srcNet1: input model 1 prototxt
        srcNet2: input model 2 prototxt
        savemodel: output combinated model caffemodel
    """
    laneNetLayers = getLayers(srcNet1)
    odNetLayers = getLayers(srcNet2)
    lane_od_net_layers = getLayers(dstNet)

    #print "\nlaneNet: ", laneNetLayers
    #print "\nodNet: ", odNetLayers
    #print "\nlane_od_net: ", lane_od_net_layers

    for index in range(0, len(lane_od_net_layers)):
        layer = lane_od_net_layers[index]
        ## shared layer, copy from odNet
        if(layer in laneNetLayers) and (layer in odNetLayers):
            copyData(dstNet, srcNet2, layer)
            print "shared layer: ", layer

        ## lane branch, copy from laneNet
        if(layer in laneNetLayers) and (layer not in odNetLayers):
            copyData(dstNet, srcNet1, layer)
            print "laneNet layer: ", layer

        ## od branch, copy from odNet
        if(layer not in laneNetLayers) and (layer in odNetLayers):
            copyData(dstNet, srcNet2, layer)
            print "odNet layer: ", layer
    
    dstNet.save(savemodel)


if __name__ == '__main__':
    init()
    combinateModel(lane_od_net, laneNet, odNet, lane_od_model)

    #copyData(lane_od_net, laneNet, 'conv4_1/bn')
   
    laneModel = '/home/yaowanchao/ADAS_caffe/fabu/models/lane/400_640_jsegnet_no_mid_upsample/finetune/ti_sim_finetune_iter_28000.caffemodel'
    lanePro = '/home/yaowanchao/ADAS_caffe/fabu/models/lane/400_640_jsegnet_no_mid_upsample/finetune/deploy_ti.prototxt'
    lane_od_pro = './lane_od_deploy.prototxt'
    
    lane_od_net = caffe.Net(lane_od_pro, lane_od_model, caffe.TEST)
    
    # Debug
    #lane_file = './laneNet.txt'
    #writeData(laneNet, lane_file)
    #lane_od_file = './lane_od_net.txt'
    #writeData(lane_od_net, lane_od_file)
    #od_file = './odNet.txt'
    #writeData(odNet, od_file)

    #printData(lane_od_net, 'conv5_1/lane_loc')
    #printData(laneNet, 'conv5_1/lane_loc')

    #checkData(lane_od_net, laneNet, 'conv1')
    #checkData(lane_od_net, laneNet, 'conv4_1/bn')

   
