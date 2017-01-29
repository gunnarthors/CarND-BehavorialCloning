import os, csv, json, math
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


def generateTrainingBatch(data, batch_size):
    while 1:    
        batch_x = np.zeros((batch_size, 66, 200, 3)) # 160, 320
        batch_y = np.zeros(batch_size)
        i = 0
        while i < batch_size:
            rint = np.random.randint(len(data)-1)
            if -0.15 < float(data[rint][1]) < 0.15:
                # Throw away some driving straight images. Only get approx 10% of them
                if np.random.randint(10) == 1:
                    batch_x[i] = getImageToBatch(data[rint][0])
                    batch_y[i] = float(data[rint][1])
                    i += 1
            else:
                batch_x[i] = getImageToBatch(data[rint][0])
                batch_y[i] = float(data[rint][1])
                i += 1

        # Randomly flip image and label
        # This will flip approx 50% of images.
        if np.randint(2) == 1:
            batch_x[i], batch_y[i] = flip(batch_x[i], batch_y[i])

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1
            )


        datagen.fit(batch_x)
        yield datagen.flow(batch_x, batch_y, batch_size=batch_size)


def flip(image, angle):
    flippedImg = cv2.flip(image,1)
    flippedAngle = angle*(-1)
    return new_image, new_angle


def getBatch(data, batch_size):
    b = generateTrainingBatch(data, batch_size)
    while 1:
        batch = next(b)
        for x, y in batch:
            yield x, y
        

def getImageToBatch(imgpath):
    return img_to_array(load_img(os.getcwd() + '/data/' + imgpath, target_size=(66,200,3))) 


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
    samples_per_epoch = batch_size * 80
    nb_epoch = 5
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
    

    # Save model.
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
    # Save weights.
    model.save_weights('model.h5')

    print("Training finished... Model and weights saved!")

if __name__ == '__main__':
    main()









