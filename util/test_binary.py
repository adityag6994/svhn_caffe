import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import sys


# Read .mat file
mat_rgb = sio.loadmat('/home/d/Desktop/model_compare_caffe/svhn/test_32x32.mat')
mat_b   = sio.loadmat('/home/d/Desktop/model_compare_caffe/svhn/test_32x32_binary.mat')

# mat_rgb = sio.loadmat('/home/d/Desktop/model_compare_caffe/svhn/train_32x32.mat')
# mat_b   = sio.loadmat('/home/d/Desktop/model_compare_caffe/svhn/train_32x32_binary.mat')

sum_rgb = 0
for i in range(0,len(mat_rgb['y'])):
	sum_rgb = sum_rgb + mat_rgb['y'][i]
print((sum_rgb))	
print(len(mat_rgb['y']))
print(mat_rgb['X'].shape)

sum_b = 0
for i in range(0,len(mat_b['y'])):
	sum_b = sum_b + mat_b['y'][i]
print((sum_b))	
print(len(mat_b['y']))
print(mat_b['X'].shape)

for i in range(0,10):
	print(mat_rgb['y'][i])
	plt.imshow(mat_rgb['X'][:,:,:,i], interpolation='nearest')
	plt.show()
	print(mat_b['y'][i])
	plt.imshow(mat_b['X'][:,:,0,i], interpolation='nearest')
	plt.show()