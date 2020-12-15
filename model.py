# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import cv2
import csv
import numpy as np
import os

def getLinesFromDrivingLogs(dataPath):
    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def findImages(dataPath):
    directories = [x[0] for x in os.walk(dataPath)]
    print(directories)
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    print(dataDirectories)
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        lines = getLinesFromDrivingLogs(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)
    
    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineImages(center, left, right, measurement, correction):
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    print(num_samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)
            
            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Convolution2D, Cropping2D
from keras.layers.convolutional import Conv2D

# Reading images locations.
centerPaths, leftPaths, rightPaths, measurements = findImages('data')
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format( len(imagePaths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imagePaths, measurements))
print('Total Sample: {}'.format( len(samples)))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
batch_size=32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Model creation
model = Sequential()
#2. data normalization
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col),output_shape=(ch, row, col)))
#3. cropping at top (70 pixel) and bottom (25pixel)
model.add(Cropping2D(cropping=((70,25), (0,0))))
#4. Nvidia architecture
model.add(Conv2D(24,(5,5),strides=(2,2), activation="relu"))
#model.add(Dropout(0.1)) ##5-2.Dropout
model.add(Conv2D(36,(5,5),strides=(2,2), activation="relu"))
#model.add(Dropout(0.1)) ##5-2.Dropout
model.add(Conv2D(48,(5,5),strides=(2,2), activation="relu"))
#model.add(Dropout(0.1)) ##5-2.Dropout
model.add(Conv2D(64,(3,3),activation="relu"))
#model.add(Dropout(0.1)) ##5-2.Dropout
model.add(Conv2D(64,(3,3),activation="relu"))
#model.add(Dropout(0.1)) ##5-2.Dropout
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3)) ##5-1.Dropout
model.add(Dense(50))
model.add(Dropout(0.3)) ##5-1.Dropout
model.add(Dense(10))
model.add(Dropout(0.3)) ##5-1.Dropout
model.add(Dense(1, name='output'))
model.summary()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,  nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

print('trained and Model saving...')
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

### plot the training and validation loss for each epoch
#import matplotlib.pyplot as plt
#print('plotting....')
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
#plt.savefig('modelAccuracy.jpg')

