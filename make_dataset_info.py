import re
import glob
from scipy.ndimage import imread
import numpy as np
from skimage.transform import resize

with open("dpchallenge_dataset.txt", 'r') as f:
	dataset_info = f.readlines()

image_files = glob.glob("images/*.jpg")
regex = r'image(\d+).jpg'
possible_small_dim = np.array([75, 80, 85, 91, 96, 102, 115, 128])

with open("dataset_info.txt", "wb") as f:
	for image_f in image_files:
		img = imread(image_f)
		dim = img.shape
		scale = max(dim) / 128.
		other_dim = min([dim[0], dim[1]]) / scale
		other_dim = possible_small_dim[(np.abs(possible_small_dim - other_dim)).argmin()]
		
		dim1 = int(dim[0] / scale)
		dim2 = int(dim[1] / scale)
		
		if min(dim1, dim2) < 50:
			continue

		if dim1 == 128:
			dim2 = other_dim
		else:
			dim1 = other_dim
	
		resized_img = resize(img, (dim1, dim2, 3))
		
		line_number = int(re.findall(regex, image_f)[0])
		info = dataset_info[line_number].split()
		num_ratings = info[2]
		average_rating = info[3]
	
		resize_file = "images_small/image" + str(line_number) + ".npy"
	
		new_info = resize_file + " " + str(resized_img.shape[0]) + " " + str(resized_img.shape[1]) + " " + str(resized_img.shape[2]) + " " + num_ratings + " " + average_rating

		np.save(resize_file, resized_img)
		f.write(new_info + "\n")
	
		if line_number % 100 == 0:
			print(line_number)	
