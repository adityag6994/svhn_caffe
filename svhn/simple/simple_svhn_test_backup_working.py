import sys, os
import caffe
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
# import helper_functions
import timeit
import lmdb
from caffe.proto import caffe_pb2
import cv2
from google.protobuf import text_format
import operator

#Initialise Caffe using GPU
caffe.set_device(0)
caffe.set_mode_gpu()
caffe.set_random_seed(0)
np.random.seed(0)
print('Initialized Caffe!')


#network_path = '/home/d/Desktop/model_compare_caffe/svhn/simple/3fcc_sigmoid_model_svhn.prototxt';
network_path = '/home/d/Desktop/model_compare_caffe/svhn/simple/svhn_simple.prototxt';
weight_path  = '/home/d/Desktop/model_compare_caffe/svhn/simple/svhn_simple.caffemodel'
net          = caffe.Net(network_path, weight_path, caffe.TEST) # caffe.TEST for testing

width  = 32
height = 32
img = cv2.imread(sys.argv[1],0)
if img.shape != [width,height]:
    img2 = cv2.resize(img,(width,height))
    img = img2.reshape(width,height,-1);
else:
    img = img.reshape(width,height,-1);
#revert the image,and normalize it to 0-1 range
img = 1.0 - img/255.0
out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))
results = list(out.values())
results = np.array(results)
print(results.shape)
print(type(results))
print(results)
max_prob = 0
max_numb = 0
for i in range(0,10):
	print(i,' -> ',results[0][0][i])
	if(max_prob < results[0][0][i]):
		max_prob = results[0][0][i]
		max_numb = i
print('-------------------------')
print(max_numb, ' ==> ', max_prob)
