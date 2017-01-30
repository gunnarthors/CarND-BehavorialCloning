import os, csv, json, math, cv2
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
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

    # Noramlization
    model.add(Lambda(normalize, input_shape=(66, 200, 3)))

    # Layer 1 - Convolutional
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))

    # Layer 2 - Convolutional
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))

    # Layer 3 - Convolutional
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))

    # Layer 4 - Convolutional
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 4 - Convolutional
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 5 Flatten
    model.add(Flatten())

    # Layer 6 - Fully-connected
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Layer 7 - Fully-connected
    model.add(Dense(50))
    model.add(Activation('relu'))

    # Layer 8 - Fully-connected
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Layer 9 - Fully-connected
    model.add(Dense(1))
    
    return model


def normalize(image):
    return image / 255.0 - 0.5


# Generate batch to train with augmentation
def generateTrainingBatch(data, batch_size):
    while 1:    
        batch_x = np.zeros((batch_size, 66, 200, 3)) # 160, 320
        batch_y = np.zeros(batch_size)
        i = 0
        while i < batch_size:
            # Get random index in data
            rint = np.random.randint(len(data)-1)
            # Get random type of image. center, left or right image
            rtype = np.random.randint(3)
            # Set offset to steerin angle if left or right image are selected
            offset = 0.0
            if rtype == 1: # Left
                offset = 0.1
            elif rtype == 2: # Right
                offset = -0.1
            # Check if steering is approx straight driving
            if -0.1 < float(data[rint][3]) < 0.1:
                # Throw away some driving straight images. Only get approx 10% of them
                if np.random.randint(10) == 1:
                    batch_x[i] = getImageToBatch(data[rint][rtype])
                    batch_y[i] = float(data[rint][3]) + offset
                    # Randomly approx 1 of 4 flipa axes and steering angle
                    if np.random.randint(4) == 1:
                        batch_x[i], batch_y[i] = flip(batch_x[i], batch_y[i])
                    i += 1
            else:
                # Other than approx straight images goes straight into batch with 1 of 4 flipping
                batch_x[i] = getImageToBatch(data[rint][rtype])
                batch_y[i] = float(data[rint][3]) + offset
                if np.random.randint(4) == 1:
                    batch_x[i], batch_y[i] = flip(batch_x[i], batch_y[i])
                i += 1

        # Some extra augmentation
        datagen = ImageDataGenerator(
            #rotation_range=5,
            #width_shift_range=0.1,
            #height_shift_range=0.1
            )

        yield datagen.flow(batch_x, batch_y, batch_size=batch_size)


# Flip images by axis and steering angle
def flip(image, angle):
    flippedImg = cv2.flip(image,1)
    flippedAngle = angle * (-1)
    return flippedImg, flippedAngle

# Yields batch from generated batch
def getBatch(data, batch_size):
    b = generateTrainingBatch(data, batch_size)
    while 1:
        batch = next(b)
        for x, y in batch:
            yield x, y
        

# Load image in size (66,200,3) and into array
def getImageToBatch(imgpath):
    return img_to_array(load_img(os.getcwd() + '/data/' + imgpath, target_size=(66,200,3))) 

# Reads the driving log csv file 
def prepareDataFromCSV(path):
    data = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([row['center'], row['left'], row['right'], row['steering']])
    return data


def main():
    path = '/data/driving_log.csv'
    training_data = prepareDataFromCSV(os.getcwd() + path)
    batch_size = 128
    samples_per_epoch = batch_size * 320
    nb_epoch = 10
    print(" Training data from csv: {}".format(path))
    print(" Batch size: {} \n Number of epochs: {} \n Samples per epoch {}"
        .format(batch_size, nb_epoch, samples_per_epoch))

    
    # To test without gpu
    #nb_epoch = 4
    #batch_size = 36
    #samples_per_epoch = 36*36

    ## Get model and start training
    model = getCNN()
    # Compile the model with adam optimizer
    adam = Adam(lr = 0.001)
    model.compile(optimizer=adam, loss="mse")

    #print(model.summary())
    ####### LOAD WEIGHTS ########
        # Load weights if they exists.
    #if os.path.isfile('model.h5'):
    #   print('Loading weights!')
    #   model.load_weights('model.h5')

    history = model.fit_generator(
        getBatch(training_data, batch_size), 
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch)
    
    
    # Save history (output from training...) 
    model.save('history.h5')

    # Save model.
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
    # Save weights.
    model.save_weights('model.h5')

    print("Training finished... Model and weights saved!")

if __name__ == '__main__':
    main()









