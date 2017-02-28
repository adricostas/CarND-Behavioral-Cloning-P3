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
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from os import getcwd
import csv
import tensorflow as tf 
from keras.utils.visualize_util import plot



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

model = network([160, 320, 3], ((50,20),(0,0)))
model.summary()
#plot(model, to_file="model.png", show_shapes=True)
