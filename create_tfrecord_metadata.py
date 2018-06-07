import os
import sys
import numpy as np
import tensorflow as tf

files = ["training.txt", "val.txt", "testing.txt"]
startstrings = ["training", "val", "testing"]

metadata = {}

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
		metadata[curr_filename] = len(shapedict[s])


np.save("metadata.npy", metadata)




