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

	# Start with noramlization
	model.add(Lambda(normalize, input_shape=(66, 200, 3)))

	# Layer 1 - Convolutional
	model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
	#model.add(BatchNormalization(epsilon=1e-06, mode=0, 
    #               axis=-1, momentum=0.99, 
    #               weights=None, beta_init='zero', 
    #               gamma_init='one'))
	model.add(Activation('relu'))

	# Layer 2 - Convolutional
	model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
	#model.add(BatchNormalization(epsilon=1e-06, mode=0, 
    #               axis=-1, momentum=0.99, 
    #               weights=None, beta_init='zero', 
    #               gamma_init='one'))
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
	model.add(Dropout(0.8))

	# Layer 6 - Fully-connected
	model.add(Dense(50))
	model.add(Activation('relu'))

	# Layer 7 - Fully-connected
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dropout(0.8))
	
	# Layer 8 - Fully-connected
	model.add(Dense(1))

	# Output layer - adam optimizer
	model.compile(optimizer="adam", loss="mse")
	
	return model


def normalize(image):
    return image / 255.0 - 0.5


def generateTrainingBatch(data, batch_size):
	batch_x = np.zeros((batch_size, 66, 200, 3))
	batch_y = np.zeros(batch_size)
	counter = 1
	while 1:
		tempnc = 0
		tempcenter = 0
		tempthrow = 0
		for b in range(batch_size):
			if float(data[b*counter][1]) != 0.0:
				batch_x[b] = getImageToBatch(data[b*counter][0])
				batch_y[b] = float(data[b*counter][1])
				tempnc += 1
				# Throw away some driving straight images
			elif np.random.randint(10) == 0:
				batch_x[b] = getImageToBatch(data[b*counter][0])
				batch_y[b] = float(data[b*counter][1])
				tempcenter += 1
			else:
				tempthrow += 1

		# Reset counter if needed. else increase by one
		if counter * batch_size  >= len(data) - 2:
			counter = 0
		else :
			counter += 1
		#print("This batch: center: {} \n not center: {} \n throw: {}".format(tempcenter, tempnc, tempthrow))
		#print(batch_y)
		yield batch_x, batch_y


def getImageToBatch(imgpath):
	return img_to_array(load_img(os.getcwd() + '/data/' + imgpath,target_size=(66,200,3)))


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
	temp = math.floor((len(training_data)/batch_size)/1.3)
	samples_per_epoch = temp * batch_size
	nb_epoch = 20
	print(" Training data from csv: {}".format(path))
	print(" Batch size: {} \n Number of epochs: {} \n Samples per epoch {}"
		.format(batch_size, nb_epoch, samples_per_epoch))

	
	# To test without gpu
	#nb_epoch = 1
	#batch_size = 3
	#samples_per_epoch = 20

	## Get model and start training
	model = getCNN()

	#x = generateTrainingBatch(training_data, batch_size)
	#next(x)
	#next(x)
	#next(x)
	#next(x)

	#print(model.summary())
	####### ADD THIS BEFORE GOING ON WITH TRAINING ########
		# Load weights if they exists.
	#if os.path.isfile('model.h5'):
	#	print('Loading weights!')
	#	model.load_weights('model.h5')
	
	history = model.fit_generator(
		generateTrainingBatch(training_data, batch_size), 
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









