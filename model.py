import os, csv, json, math
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import np_utils

import matplotlib.pyplot as plt

# Return the model. 
# Working with Nvidia network architecture with some tweaks.
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def getCNN():
	
	model = Sequential()
	
	def resize(image):
		import tensorflow as tf
		return tf.image.resize_images(image, (66, 200))
	# Resize Image
	model.add(Lambda(resize, input_shape=(160,320, 3)))

	# Noramlization
	model.add(Lambda(normalize)) #, input_shape=(66, 200, 3)

	# Layer 1 - Convolutional
	model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
	model.add(Activation('relu'))

	# Layer 2 - Convolutional
	model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
	model.add(Activation('relu'))

	# Layer 3 - Convolutional
	model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(epsilon=1e-06, mode=0, 
    #               axis=-1, momentum=0.99, 
    #               weights=None, beta_init='zero', 
    #               gamma_init='one'))


	# Layer 4 - Convolutional
	model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(epsilon=1e-06, mode=0, 
    #               axis=-1, momentum=0.99, 
    #               weights=None, beta_init='zero', 
    #               gamma_init='one'))

	# Layer 4 - Convolutional
	model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
	model.add(Activation('relu'))
	#model.add(BatchNormalization(epsilon=1e-06, mode=0, 
    #               axis=-1, momentum=0.99, 
    #               weights=None, beta_init='zero', 
    #               gamma_init='one'))

	#  Flatten
	model.add(Flatten())

	# Layer 5 - Fully-connected
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# Layer 6 - Fully-connected
	model.add(Dense(50))
	model.add(Activation('relu'))

	# Layer 7 - Fully-connected
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	# Layer 8 - Fully-connected
	model.add(Dense(1))
	
	return model


def normalize(image):
    return image / 255.0 - 0.5


# def generateTrainingBatch(data, batch_size):
# 	batch_x = np.zeros((batch_size, 66, 200, 3))
# 	batch_y = np.zeros(batch_size)
# 	counter = 1
# 	while 1:
# 		for b in range(batch_size):
# 			if float(data[b*counter][1]) != 0.0:
# 				batch_x[b] = getImageToBatch(data[b*counter][0])
# 				batch_y[b] = float(data[b*counter][1])
# 				# Throw away some driving straight images
# 			elif np.random.randint(10) == 0:
# 				batch_x[b] = getImageToBatch(data[b*counter][0])
# 				batch_y[b] = float(data[b*counter][1])

# 		# Reset counter if needed. else increase by one
# 		if counter * batch_size  >= len(data) - 2:
# 			counter = 0
# 		else :
# 			counter += 1

# 		yield batch_x, batch_y

def generateTrainingBatch(data, batch_size):
	s = 0
	while 1:	
	 	batch_x = np.zeros((batch_size, 160, 320, 3)) # 66, 200
	 	batch_y = np.zeros(batch_size)
	 	i = 0
	 	while i < batch_size:
	 		rint = np.random.randint(len(data)-1)

	 		if float(data[s][1]) != 0.0:
	 			batch_x[i] = getImageToBatch(data[s][0])
	 			batch_y[i] = float(data[s][1])
	 			i += 1
	 		# Throw away some driving straight images
	 		elif np.random.randint(10) == 0:
	 			batch_x[i] = getImageToBatch(data[s][0])
	 			batch_y[i] = float(data[s][1])
	 			i += 1
	 		# Resetting data counter if bigger than data
	 		if s >= len(data) - 1:
	 			s = 0
	 		else:
	 			s += 1

	 	datagen = ImageDataGenerator(
	    	featurewise_center=True,
	    	featurewise_std_normalization=True,
	    	rotation_range=10,
	    	width_shift_range=0.2,
	    	height_shift_range=0.2)

	 	datagen.fit(batch_x)
	 	yield datagen.flow(batch_x, batch_y, batch_size=batch_size)

def getBatch(data, batch_size):
	b = generateTrainingBatch(data, batch_size)
	while 1:
		batch = next(b)
		for x, y in batch:
			yield x, y

def getImageToBatch(imgpath):
	return img_to_array(load_img(os.getcwd() + '/data/' + imgpath)) #, target_size=(66,200,3)


def prepareDataFromCSV(path):
	data = []
	with open(path) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append([row['center'], row['steering']])
	return data


def main():
	path = '/data/driving_log.csv'
	training_data = prepareDataFromCSV(os.getcwd() + path)
	batch_size = 128
	samples_per_epoch = batch_size * 120
	nb_epoch = 5
	print(" Training data from csv: {}".format(path))
	print(" Batch size: {} \n Number of epochs: {} \n Samples per epoch {}"
		.format(batch_size, nb_epoch, samples_per_epoch))

	
	# To test without gpu
	nb_epoch = 1
	batch_size = 5
	samples_per_epoch = 20

	## Get model and start training
	model = getCNN()
	# Compile the model with adam optimizer
	model.compile(optimizer="adam", loss="mse")

	#print(model.summary())
	####### ADD THIS BEFORE GOING ON WITH TRAINING ########
		# Load weights if they exists.
	#if os.path.isfile('model.h5'):
	#	print('Loading weights!')
	#	model.load_weights('model.h5')

	history = model.fit_generator(
		getBatch(training_data, batch_size), 
		samples_per_epoch=samples_per_epoch,
		nb_epoch=nb_epoch)
	

	# Save model.
	json_string = model.to_json()
	with open('model.json', 'w') as outfile:
		json.dump(json_string, outfile)
	# Save weights.
	model.save_weights('model.h5')

	print("Training finished... Model and weights saved!")

if __name__ == '__main__':
	main()









