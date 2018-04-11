import sys, os
import caffe
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import lmdb

def image_generator(db_path):
    """yeild normalised images"""
    db_handle = lmdb.open(db_path, readonly=True)
    with db_handle.begin() as db:
        cur = db.cursor()
        for _, value in cur:
            #Read the LMDB and transform into numpy array
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            int_x = caffe.io.datum_to_array(datum)
            x = np.asfarray(int_x, dtype=np.float32)
            yield x - 128

def batch_generator(shape, db_path):
    gen = image_generator(db_path)
    res = np.zeros(shape)
    while True:
        for i in range(shape[0]):
            res[i] = next(gen)
        yield res
        
#Testing the netwrok
def test_network(test_net, db_path_test):
    accuracy = 0;
    loss = 0 ;
    test_batches = 0;
    input_shape = test_net.blobs["data"].data.shape;
    for test_batch in batch_generator(input_shape, db_path_test):
        test_batches += 1;
        test_net.blobs["data"].data[...] = test_batch;
        test_net.forward();
        #collect the output
        accuracy += test_net.blobs["accuracy"].data;
        loss += test_net.blobs["loss"].data;
    return (accuracy / test_batches, loss / test_batches)

def print_network(net):
    #Print Strucuture of current Network used
    print ('Network Structure : ')
    for name, layers in zip(net._layer_names, net.layers):
        print("{:<7}: {:17s}({} blobs)".format(name, layers.type, len(layers.blobs)))
        print ('|| ')
        print("Blobss:")
    for name, blob in net.blobs.items():
        print("{:<5}:  {}".format(name, blob.data.shape)) 