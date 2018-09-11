import argparse
import os
import cv2
import numpy as np
import caffe
from lane_process_adv import laneProcess

FLAGS = None

class EvalLane():
    def __init__(self, pro, model, imglist, respath, gpu):
        self.prototxt = pro
        self.model = model
        self.imglist = imglist
        self.path = respath
        self.gpu = gpu
        self.labels = {0:"back", 1: "l1", 2:"r1", 3:"l2", 4:"r2", 5:"m"}
        print 'Init eval lane'

    def initNet(self):
        if self.gpu:
            caffe.set_device(self.gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(self.prototxt, self.model, caffe.TEST)

        self.mu = np.array([104.0, 117.0, 123.0])
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', self.mu)
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_input_scale('data', 0.007843)
        self.transformer.set_channel_swap('data', (2, 1, 0))

    def parseImagelist(self):
        allImg = []
        with open(self.imglist, 'r') as f:
            lines = f.readlines()
            for imgId in range(0, len(lines)):
                line = lines[imgId].strip('\n')
                if not os.path.exists(line):
                    pass
                else:
                    allImg.append(line)
        return allImg
 
    def softmax(self, x, axis):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x, axis=0)
        return softmax_x

    def sigmoid(self, x):
        s = None
        s = 1 / (1 + np.exp(-x))
        return s

    def eval(self):
        if not os.path.exists(self.imglist):
            print 'imglist not exists.'
            exit(0)

        self.initNet()

        lines = self.parseImagelist()
        for i in range(0, len(lines)):
            line = lines[i].strip('\n')
            self.imgName = line.split('/')[-1]
           
            try:
                img = caffe.io.load_image(line)
            except IOError:
                print "Open failed: ", line
            else: 
                transformed_image = self.transformer.preprocess('data', img)
                self.net.blobs['data'].data[...] = transformed_image
                output = self.net.forward()

                #score = output['conv5_1/lane_loc'][0]
                score = self.net.blobs['ctx_final'].data[0]
                score = self.softmax(score, axis=0)
                score[np.where(score < 0.6)] = 0
                classed = np.argmax(score, axis=0)
                #print "score shape: ", score.shape
                
                #classed = output["argMaxOut"][0][0]

                classed = classed.astype(np.int)
                labels = [self.labels[s] for s in np.unique(classed)]
                print "labels: ", np.unique(classed), labels
                
                img = img * 255
                img = img[..., [2, 1, 0]]
                img = cv2.resize(img, (640, 400), interpolation=cv2.INTER_LINEAR)
                # ratio: the ratio between input image size and output feature map size
                laneprocess = laneProcess(self.path, ratio = 8)
                all_ext_points = laneprocess.processLane(classed, img, self.imgName, drawFlag=True)
                print "all_ext_points: ", all_ext_points
                
                #print "imageName: ", self.imgName
                cv2.imwrite(self.path + self.imgName + "_res.png", classed*32)
                cv2.imwrite(self.path + self.imgName + "_src.jpg", img)

                #assert(i < 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-d', "--deploy",
            type = str,
            required = True,
            default = 'deploy.prototxt',
            help = "The deploy prototxt needed in eval.")

    parser.add_argument(
            '-m', '--model',
            type = str,
            required = True,
            default = 'test.caffemodel',
            help = 'The caffemodel needed in eval.')
    parser.add_argument(
            '-i', "--image",
            type = str,
            required = True,
            default = 'test.txt',
            help = 'The test image list.')
    parser.add_argument(
            '-p', '--path',
            type = str,
            required = True,
            default = './result/',
            help = "The path to save test result."
            )
    parser.add_argument(
            '-g', '--gpu',
            type = int,
            default = 1,
            help = 'Switch for gpu computation.')
    FLAGS, _ = parser.parse_known_args()
    evalTest = EvalLane(FLAGS.deploy, FLAGS.model, FLAGS.image, FLAGS.path, FLAGS.gpu)
    evalTest.eval()
