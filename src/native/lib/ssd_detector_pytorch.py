
import sys
from os import path
#print(sys.path)
import numpy as np

import cv2
import time
import torch
from torch.autograd import Variable
sys.path.append(path.dirname(path.abspath(__file__))+
    '/ssd.pytorch')
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

class SSDPyTorch:
    def __init__(self, weights_filename):
        if type(weights_filename)!=type('str'):
            weights_filename = weights_filename.decode("utf-8")
        is_existed = path.exists(weights_filename)
        if not is_existed:
            print("Cannot find weights file...")
            return -1
        self.net = build_ssd('test', 300, 21)
        self.net.load_state_dict(torch.load(weights_filename))
        self.transform = BaseTransform(self.net.size, 
            (104/256.0, 117/256.0, 123/256.0))

    def detect(self, frame):
        net = self.net
        transform = self.transform
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2,0,1)
        x = Variable(x.unsqueeze(0))
        print("Before reference ...")
        y = net(x) # foward pass
        print("After reference ...")
        detections = y.data
        print("After y.data ...")
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        print("After scale ...")
        bbs = []
        conf= []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                bbs.append(pt)
                conf.append(detections[0,i,j,0])
                cv2.rectangle(frame,
                             (int(pt[0]), int(pt[1])),
                             (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        bbs = np.asarray(bbs)
        conf= np.asarray(conf)
        print(bbs)
        print(conf)
        return (bbs, conf)

    def run_v2(self, filename):
        filename = filename[0]
        filename = filename.decode("utf-8")
        img = cv2.imread(filename)
        #out = detect(net, transform, img)
        #cv2.imwrite('res.png', out)
        print('Get image successfully.')
        bbs, conf = self.detect(img)
        print ('Detect DONE!')
        return (bbs, conf)
        #return bbs

    def run_v3(self, input_data, w, h, c):
        print('Come in run_v3 ...')
        print(input_data)
        print(type(input_data[0][0][0]))
        print(input_data.shape)

    def run_v4(self, frame):
        #print(type(frame))
        frame = frame[0]
        print(frame.shape)
        out, score = self.detect(frame)
        print("In run_v4, Done!")
        return (out, score)
