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
from PIL import Image

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
min_range = 0 
max_range = 26000
data = '/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images/'
result = '/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images.txt'
mean = '/home/d/Desktop/model_compare_caffe/svhn/simple/svhn_simple.npy'

f  = open(result, "r")
fs = f.read()
words  = fs.split()
number = [int(w) for w in words]
accuracy = 0

mean_image = np.load(mean)
print(mean_image)
print('Mean Image : ', mean_image.shape, ' ', type(mean_image))

for current_image in range(min_range, max_range):
	max_prob = 0
	max_numb = 0
	path = data + str(current_image) + '.png'
	img = cv2.imread(path,0)
	
	#Print the accuracy in between
	if(current_image%1000 == 1):
		print('Current Image : ', current_image , ' | Accuracy :  ' ,(accuracy/current_image)*100 )
		
	#Change the shape of the image
	if img.shape != [width,height]:
	    img2 = cv2.resize(img,(width,height))
	    img = img2.reshape(width,height,-1);
	else:
	    img = img.reshape(width,height,-1);
	
	#Revert the image,and normalize it to 0-1 range
	img = 1.0 - img/255.0
	mean_image = 1.0 - mean_image/255
	# With mean image
	out = net.forward_all(data=np.asarray([img.transpose(2,0,1) - mean_image]))
	# Without mean image 
	# out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))

	# Calculate Accuracy
	results = np.array(list(out.values()))
	if(np.argmax(results) == number[current_image]):
		accuracy = accuracy+1;

# for current_image in range(min_range, max_range):
# 	max_prob = 0
# 	max_numb = 0
# 	path = data + str(current_image) + '.png'
# 	img = cv2.imread(path,0)
# 	if(current_image%1000 == 0):
# 		print(current_image)
# 	if img.shape != [width,height]:
# 	    img2 = cv2.resize(img,(width,height))
# 	    img = img2.reshape(width,height,-1);
# 	else:
# 	    img = img.reshape(width,height,-1);
# 	#revert the image,and normalize it to 0-1 range
# 	img = 1.0 - img/255.0
# 	out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))
# 	results = list(out.values())
# 	results = np.array(results)
# 	# print(np.argmax(results),' <-> ', number[current_image])
# 	if(np.argmax(results) == number[current_image]):
# 		accuracy = accuracy+1;

# img = cv2.imread('/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images/0.png',0)
# print(type(img))
# if img.shape != [width,height]:
#     img2 = cv2.resize(img,(width,height))
#     img = img2.reshape(width,height,-1);
# else:
#     img = img.reshape(width,height,-1);
# #revert the image,and normalize it to 0-1 range
# img = 1.0 - img/255.0
# out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))
# results = list(out.values())
# results = np.array(results)
# print(results.shape)
# print(type(results))
# print(results)
# for i in range(0,10):
# 	print(i,' -> ',results[0][0][i])
# 	if(max_prob < results[0][0][i]):
# 		max_prob = results[0][0][i]
# 		max_numb = i
# print('-------------------------')
# print(max_numb, ' ==> ', max_prob)

print('Accuracy :', accuracy/(max_range-min_range)*100)
