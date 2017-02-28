from sklearn.utils import shuffle
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from os import getcwd
import csv
import tensorflow as tf 

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def load_data (paths):
	images_paths = []
	steering_angles = []

	for file_path in paths:
		with open(file_path, 'r') as f:
			#skip first line - heading
			
			reader = csv.reader(f, skipinitialspace = True, delimiter = ',')

			for row in reader:
				if isfloat(row[3]) == False:
					next(f, None)
					continue
				steering_center = float(row[3])
				# create adjusted steering measurements for the side camera images
				correction = 0.2 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				# read in images from center, left and right cameras
				# fill in the path to your training IMG directory
				if row[0][0:2] == 'IMG':
					root_path = getcwd() + '/'
				else: 
					root_path = ''
				img_center = root_path + row[0]
				img_left = root_path + row[1]
				img_right = root_path + row[2]
			
				#add images and angles to data set
				images_paths.extend((img_center, img_left, img_right))
				steering_angles.extend((steering_center, steering_left, steering_right))

	images_paths = np.array(images_paths)
	steering_angles = np.array (steering_angles)

	return images_paths, steering_angles


csv_files = ['./my_own_data/driving_log.csv', './my_own_data_track2/driving_log.csv']
#csv_files = ['./driving_log.csv', './my_own_data/driving_log.csv']
images_paths, steering_angles = load_data (csv_files)

print('Total number of images is: ', images_paths.shape)

num_bins = 100
avg_samples_per_bin = len(steering_angles)/num_bins
hist, bins = np.histogram(steering_angles, num_bins)
f = plt.figure(1)
plt.hist(steering_angles, num_bins)
plt.plot((np.min(steering_angles), np.max(steering_angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
f.show()

keep_probs = []
target = avg_samples_per_bin 
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(steering_angles)):
    for j in range(num_bins):
        if steering_angles[i] > bins[j] and steering_angles[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
images_paths = np.delete(images_paths, remove_list, axis=0)
steering_angles = np.delete(steering_angles, remove_list)

hist, bins = np.histogram(steering_angles, num_bins)
g = plt.figure(2)
plt.hist(steering_angles, num_bins)
plt.plot((np.min(steering_angles), np.max(steering_angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
g.show()

input()