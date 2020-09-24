# IrisTL-PAD solution
# Created by Cunjian Chen (cunjian@msu.edu)
from ctypes import *
import math
import random
import glob
import cv2
import os
import pickle
import sys
import datetime
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res
    
if __name__ == "__main__":

    net_det = load_net(b"cfg/detection.cfg", b"backup_det/yolo-final.weights", 0)
    meta_det = load_meta(b"data/iris_det.data")
    net = load_net(b"cfg/classification.cfg", b"backup_crop/classification_60.weights", 0)
    meta = load_meta(b"data/LivDet-crop.data")

    test_filenames=glob.glob(sys.argv[1]+'/*.png')
    for filename in test_filenames:
        print(filename)
        path, img_filename = os.path.split(filename)
        # Switch the filename and save the score
        score_filename=img_filename
        score_filename=score_filename[:-4]+'.txt'
        textfile_score=open(path+'/'+score_filename,'w')
        r = detect(net_det, meta_det, bytes(filename,encoding='utf-8'))
        if not r:
            pd_score=1 
            textfile_score.write("%s\n" % pd_score)
        else:
            box_iris=r[0][2]
            img=cv2.imread(filename)
            height, width, channels = img.shape
        
            center_x=int(box_iris[0])
            center_y=int(box_iris[1])
            box_width=int(box_iris[2])
            box_height=int(box_iris[3])  
            crop_im = img[center_y-int(box_height/2):center_y+int(box_height/2), center_x-int(box_width/2):center_x+int(box_width/2)]
            cv2.imwrite('Evaluation/temp.png',crop_im)
            im = load_image(b'Evaluation/temp.png', 0, 0)
            r = classify(net, meta, im)

            if r[0][0]=='Live':   
                pd_score=1-r[0][1]
                textfile_score.write("%s\n" % (1-r[0][1]))
            else:
                pd_score=r[0][1]
                textfile_score.write("%s\n" % r[0][1])
     
        textfile_score.close()   
