import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


prototxt_solver = "/home/d/Desktop/model_compare_caffe/3fcc_sigmoid_solver.prototxt"
solver_config = caffe_pb2.SolverParameter()
with open(prototxt_solver) as f:
	text_format.Merge(str(f.read()), solver_config)

# print(solver_config.test_interval)
# solver_config.test_interval = 1000
# print(solver_config.test_interval)


layer_size = ['inv', 'fixed', 'poly', 'step']
for i in layer_size:
	print(i)
	temp_solver_config = solver_config
	temp_solver_config.lr_policy = i
	temp_solver_config = text_format.MessageToString(temp_solver_config)
	with open('temp.prototxt','w') as f:
		f.write(temp_solver_config)
