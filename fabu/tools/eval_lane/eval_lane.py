import argparse
import os
import cv2
import shutil
import numpy as np
import caffe

FLAGS = None

class EvalLane():
    def __init__(self, pro, model, imglist, respath, gpu):
        self.prototxt = pro
        self.model = model
        self.imglist = imglist
        self.path = respath
        self.gpu = gpu
        self.labels = {0:"back", 1: "l1", 2:"r1", 3:"l2", 4:"r2", 5:"mid"}
        print 'Init eval lane'

    def rescore(self, scores, c):
        return np.where(scores == c)[0][0]

    def eval(self):
        if self.gpu:
            caffe.set_device(self.gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        net = caffe.Net(self.prototxt, self.model, caffe.TEST)

        mu = np.array([104.0, 117.0, 123.0])
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_input_scale('data', 0.007843)
        transformer.set_channel_swap('data', (2, 1, 0))

        if not os.path.exists(self.imglist):
            print 'imglist not exists.'
            exit(0)

        print 'start'
        with open(self.imglist, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines)):
                line = lines[i].strip('\n')
                print line
                imgName = line.split('/')[-1]
                try:
                    img = caffe.io.load_image(line)
                except IOError:
                    pass
                else:
                    transformed_image = transformer.preprocess('data', img)
                    net.blobs['data'].data[...] = transformed_image
                    output = net.forward()

                    score = output['conv5_1/lane_loc'][0]
                    #score = output['out_deconv_final_up4'][0]
                    #score = output['ctx_softmax'][0]
                    print score.shape

                    classed = np.argmax(score, axis=0)

                    # softmax conf >= 0.4
                    '''
                    for i in range(0, score.shape[1]):
                        for j in range(score.shape[2]):
                            classed[i, j] = 0
                            for cls in range(score.shape[0]):
                                if score[cls, i, j] >= 0.4:
                                    classed[i, j] = cls
                    '''

                    names = dict()
                    all_labels = self.labels
                    scores = np.unique(classed)
                    labels = [all_labels[s] for s in scores]
                    print labels
                    
                    painted = classed * 32
                    print np.where(painted == 64)
                    tmp = np.unique(painted)
                    if (32 in tmp) or (64 in tmp) or (96 in tmp) or (128 in tmp) or (160 in tmp):
                        print line
                    print painted.shape
                    cv2.imwrite(self.path + imgName, painted)
                    shutil.copy(line, self.path + imgName + "_src.jpg")


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
