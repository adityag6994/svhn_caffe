# import sys, os
# import numpy
# import caffe
# import lmdb
# import random
# from caffe.proto import caffe_pb2
# import scipy.io as sio
# from matplotlib import pyplot as plt


# os.chdir("/home/d/Desktop/model_compare_caffe/svhn")
# SVHN_train_RGB = sio.loadmat('train_32x32.mat', squeeze_me=True, struct_as_record=False)
# SVHN_test_RGB = sio.loadmat('test_32x32.mat', squeeze_me=True, struct_as_record=False)
# print(type(SVHN_train_RGB))
# print(SVHN_train_RGB['X'].shape)


# SVHN_train = (numpy.sum(SVHN_train_RGB['X'],2)/3)
# SVHN_test = (numpy.sum(SVHN_test_RGB['X'],2)/3)
# print(type(SVHN_train))
# print(SVHN_train['X'].shape)

# use case :  python convert_channel.py ../svhn/test_32x32.mat 

import sys
import numpy as np
import caffe
import lmdb
import random
from caffe.proto import caffe_pb2
import scipy.io as sio
import scipy.misc
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print ("Usage: python convert_mat_to_lmdb.py [file.mat]")

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

result = []

# Read .mat file
mat = sio.loadmat(sys.argv[1])
print('Input Shape : ', mat['X'].shape)
a=mat['X'].shape
test_size = (32,32,a[3])
mat_b = np.zeros(test_size)


#save the text file filled with correct results
f = open('/home/d/Desktop/model_compare_caffe/svhn/svhn_test_images/svhn_test_images.txt','w')
print('file opened for writing ...')

for i in range(0,len(mat['y'])):
	f.write(str(int(mat['y'][i])) + '\n')

print('closing the file ...')
f.close()
exit()

for i in range(0,mat['X'].shape[3]):
	# plt.imshow(mat['X'][:,:,:,i])
	# plt.show()
	mat_gray_current = rgb2gray(mat['X'][:,:,:,i])
	mat_b[:,:,i] = mat_gray_current
	# To save it to computer
	#plt.imshow(mat_gray_current)
	#plt.show()
	#path = '/home/d/Desktop/model_compare_caffe/svhn_dataset/' + str(i) + '.png'
	# path = str(i) + '.png'
	#print(path)
	#scipy.misc.imsave(path, mat_gray_current)

exit()

mat_b = np.expand_dims(mat_b, axis=2)
print('Output Shape : ', mat_b.shape)

for i in range(0,10):
	print(mat['y'][i])

exit()
#save to new .mat file
dict_temp = {}
dict_temp['y'] = mat['y']
dict_temp['X'] = mat_b
# sio.savemat('/home/d/Desktop/model_compare_caffe/svhn/test_32x32_binary.mat',mdict=dict_temp)
sio.savemat('/home/d/Desktop/model_compare_caffe/svhn/train_32x32_binary.mat',mdict=dict_temp)



