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


# Flip images by axis and steering angle
def flip(image, angle):
    flippedImg = cv2.flip(image,1)
    flippedAngle = angle * (-1)
    return flippedImg, flippedAngle

# Normalize images
def normalize(image):
    return image / 255.0 - 0.5

# Load image from path to np array. CV2 loads image in BGR but we change it to RGB as well.
def getImageToBatch(imgpath):
    img = cv2.imread(imgpath)
    return img

# Change brightness for more generalization(day,night,dusk,dawn etc...)
def randomBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = hsv[:,:,2]*(1.0 + np.random.uniform(-.8, .5))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Crop 60pixels of the top of the image as they are not needed for driving..
# and we take 20px from bottom to get rid of the car from the image
def cropTopBot(img):
    return img[60:140, 0:320] # Crop from x, y, w, h 

# Resize the image to fit our model input shape (66,200,3)
def resizeImg(img):
    return cv2.resize(img, (200,66))

# Reads the driving log csv file 
def prepareDataFromCSV(path):
    data = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(
                [
                    os.getcwd() + '/data/' + row['center'], 
                    os.getcwd() + '/data/' + row['left'], 
                    os.getcwd() + '/data/' + row['right'], 
                    row['steering']
                ])
    return data


def getBatch(data, batch_size):
    steeringIndex = 3
    while 1:
        batch_x = np.zeros((batch_size, 66, 200, 3))
        batch_y = np.zeros(batch_size)
        i = 0
        while i < batch_size:
            useImg = False
            # Get random index in data and set steering value accordingly
            rint = np.random.randint(len(data)-1)
            steeringValue = float(data[rint][steeringIndex])

            # Check if steering is approx straight driving - We dont want to take them all...
            if -0.1 <= steeringValue <= 0.1:
                if np.random.randint(10) == 1:
                    useImg = True

            # All images which are not as near 0.0 we will use
            else:
                useImg = True

            if useImg:
                # Get random type of image. center, left or right image
                rtype = np.random.randint(3)

                if rtype == 1: # Left image add offset
                    steeringValue += 0.2
                if rtype == 2: # Right image add offset
                    steeringValue -= 0.2

                batch_y[i] = steeringValue
                batch_x[i] = resizeImg(cropTopBot(randomBrightness(getImageToBatch(data[rint][rtype]))))

                # Add random flip by axes images. Approx 1 of 5
                if np.random.randint(2) == 1:
                    batch_x[i], batch_y[i] = flip(batch_x[i], batch_y[i])

                # As we used the image i will increse by one
                i += 1

        yield batch_x, batch_y



def main():
    path = '/data/driving_log.csv'
    training_data = prepareDataFromCSV(os.getcwd() + path)
    batch_size = 128
    samples_per_epoch = batch_size * 75
    nb_epoch = 25
    print(" Training data from csv: {}".format(path))
    print(" Batch size: {} \n Number of epochs: {} \n Samples per epoch {}"
        .format(batch_size, nb_epoch, samples_per_epoch))


    # To test without gpu
    #nb_epoch = 4
    #batch_size = 32
    #samples_per_epoch = 36*10

    ## Get model and start training
    model = getCNN()
    # Compile the model with adam optimizer
    adam = Adam(lr = 0.0001)
    model.compile(optimizer=adam, loss="mse")

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









