import caffe
import lmdb
import sys
import numpy as np
from caffe.proto import caffe_pb2

if len(sys.argv) != 2:
    print ("Usage: python convert_mat_to_lmdb")


lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print('{},{}'.format(key, label))

