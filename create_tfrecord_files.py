import os
import sys
import numpy as np
import tensorflow as tf

files = ["training.txt", "val.txt", "testing.txt"]
startstrings = ["training", "val", "testing"]

print("hello")

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for i, fname in enumerate(files):
	
	shapedict = {}
	with open(fname) as f:
		for line in f.readlines():
			data = line.split()
			shape = (data[1], data[2])
			vals = (data[0], data[5])

			if shape in shapedict.keys():
				shapedict[shape].append(vals)
			else:
				shapedict[shape] = [vals]

	for s in shapedict.keys():
		curr_filename = startstrings[i] + "_" + str(s[0]) + "_" + str(s[1])
		writer = tf.python_io.TFRecordWriter("tf_files/" + curr_filename + ".tfrecords")
		for img_name, rating in shapedict[s]:
		    
			img = np.load(img_name)
		    
			feature = {curr_filename + '/label': _float_feature(float(rating)), curr_filename + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
			example = tf.train.Example(features=tf.train.Features(feature=feature))  
			writer.write(example.SerializeToString())
		    
		writer.close()
		sys.stdout.flush()




