from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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
				#Check if we are at the "title" row. It's necessary when we work with udacity data.
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
				#This "if" is necessary when we use udacity data.
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


# def generator (images_paths, angles, batch_size = 128):
# 	'''Method for the model training data generator to load and process 
# 	images and then yield them to the model'''
# 	num_samples = len(images_paths)
# 	while True:		
# 		for offset in range(0, num_samples, batch_size):
# 			X,y = ([],[])
# 			images_paths, angles = shuffle(images_paths, angles)
# 			batch_x, batch_y = images_paths[offset : offset + batch_size], angles[offset : offset + batch_size]
# 			for i in range(len(batch_x)):
# 				img = np.asarray(Image.open(images_paths[i]))
# 				#img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# 				X.append(img)
# 				y.append(angles[i])
			
# 			yield (np.array(X), np.array(y))
			
def generator(images_paths, angles, batch_size=128):
    '''
    Method for the model training data generator to load images and then yield them to the
    model. '''
    images_paths, angles = shuffle(images_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = np.asarray(Image.open(images_paths[i]))
            angle = angles[i]
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                images_paths, angles = shuffle(images_paths, angles)

            # data augmentation
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if abs(angle) > 0.33:
                img = cv2.flip(img, 1)
                angle *= -1
                X.append(img)
                y.append(angle)
                if len(X) == batch_size:
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    images_paths, angles = shuffle(images_paths, angles)

def network (input_shape, crop_shape):
	
	model = Sequential()

	#Crop image
	model.add(Cropping2D(crop_shape, input_shape = input_shape, name= 'Crop'))
	# Normalize
	model.add(BatchNormalization(axis=1, name="Normalize"))

	# Add three 5x5 convolution layers (output depth 24,36,48) each with 2x2 stride
	model.add(Convolution2D(24, 5, 5, subsample =(2,2), border_mode ='valid', 
		W_regularizer = l2(0.001), name = 'Convolution2D1'))
	model.add(ELU())
	model.add(Convolution2D(36, 5, 5, subsample = (2,2), border_mode = 'valid', 
		W_regularizer = l2(0.001), name = 'Convolution2D2'))
	model.add(ELU())
	model.add(Convolution2D(48, 5, 5, subsample = (2, 2), border_mode = 'valid',
		W_regularizer = l2(0.001), name = 'Convolution2D3'))

	
	# Add two 3x3 convolution layers (output depth 64, and 64)
	model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001), 
		name = 'Convolution2D4'))
	model.add(ELU())
	model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001),
		name = 'Convolution2D5'))
	model.add(ELU())

	# Add a flatten layer
	model.add(Flatten(name = 'Flatten'))

	# Add three fully connected layers (depth 100, 50, 10)
	model.add(Dense(100, W_regularizer=l2(0.001), name = 'FC2'))
	model.add(ELU())

	model.add(Dense(50, W_regularizer=l2(0.001), name = 'FC3'))
	model.add(ELU())
	
	model.add(Dense(10, W_regularizer=l2(0.001), name = 'FC4'))
	model.add(ELU())
	

	# Add a fully connected output layer
	model.add(Dense(1, name = 'Readout'))

	return model


###################    Main program   ######################################

#csv_files = ['./driving_log.csv', './my_own_data/driving_log.csv']
csv_files = ['./my_own_data/driving_log.csv', './my_own_data_track2/driving_log.csv']

images_paths, steering_angles = load_data (csv_files)

batch_size = 128

print('Total number of images is: ', images_paths.shape)

num_bins = 25
avg_samples_per_bin = len(steering_angles)/num_bins
hist, bins = np.histogram(steering_angles, num_bins)

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

#Shuffle before split into train and validation set
images_paths, steering_angles = shuffle(images_paths, steering_angles)

#split into train/validation set
images_paths_train, images_paths_valid, steering_angles_train, steering_angles_valid = train_test_split(images_paths, 
	steering_angles, test_size = 0.2, random_state = 42)

print('Train:', images_paths_train.shape, steering_angles_train.shape)
print('Valid:', images_paths_valid.shape, steering_angles_valid.shape)


model = network([160, 320, 3], ((50,20),(0,0)))

model.summary()

model.compile(loss="mse", optimizer="adam")

train_generator = generator (images_paths_train, steering_angles_train)
valid_generator = generator (images_paths_valid, steering_angles_valid)

history = model.fit_generator(train_generator, validation_data = valid_generator,
	nb_val_samples = len(steering_angles_valid), samples_per_epoch = len(steering_angles_train),
	nb_epoch = 20, verbose = 1)



# Save model data
model.save('./model.h5')
print('Model saved')


### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
