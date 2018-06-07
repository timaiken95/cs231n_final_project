from random import shuffle

with open("dataset_info.txt", 'r') as f:
	lines = f.readlines()
	shuffle(lines)
	training = lines[:12000]
	testing = lines[12000:14000]
	val = lines[14000:]

	with open("training.txt", 'wb') as tr:
		for l in training:
			tr.write(l)

	with open("testing.txt", 'wb') as te:
		for l in testing:
			te.write(l)

	with open("val.txt", 'wb') as va:
		for l in val:
			va.write(l)

	print("Done")
