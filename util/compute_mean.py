import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import time
import sys
import os

if len(sys.argv) != 2:
    print ("Usage: python convert_mat_to_lmdb.py [lmdb folder]")


lmdb_env=lmdb.open(sys.argv[1])
lmdb_txn=lmdb_env.begin()
lmdb_cursor=lmdb_txn.cursor()
datum=caffe_pb2.Datum()

N=0
mean = np.zeros((1, 1, 28, 28))
begintime = time.time()

for key,value in lmdb_cursor:
    datum.ParseFromString(value)
    data=caffe.io.datum_to_array(datum)
    image=data.transpose(1,2,0)
    # print(type(image))
    # print(image.shape)
    # exit()
    mean[0,0] += image[:, :, 0]
    #mean[0,1] += image[:, :, 1]
    #mean[0,2] += image[:, :, 2]
    N+=1
    if N % 1000 == 0:
        elapsed = time.time() - begintime
        print("Processed {} images in {:.2f} seconds. "
              "{:.2f} images/second.".format(N, elapsed,
                                             N / elapsed))
mean[0]/=N
blob = caffe.io.array_to_blobproto(mean)
os.umask(0)
with open('mean_images/mean_test.binaryproto', 'wb') as f:
    f.write(blob.SerializeToString())

lmdb_env.close()