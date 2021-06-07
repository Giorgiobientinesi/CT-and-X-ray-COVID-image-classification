import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time



Data_directory = "COVID-19 Dataset/CT" #Same for "COVID-19 Dataset/X-ray"

Categories = ["COVID", "Non-COVID"]


Size_of_images = 150




training_data = []
def training_creation():
    for category in Categories:
        path = os.path.join(Data_directory, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Size_of_images, Size_of_images), cv2.INTER_AREA)
            training_data.append([new_array,class_num])

training_creation()

print(len(training_data))

random.shuffle(training_data)


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, Size_of_images, Size_of_images, 1)
y = np.array(y)


X = X/255.0

dense_layers = [1]
layer_sizes = [128]  #64, 128
conv_layers = [3] #3

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=64,
                      epochs=15,
                      validation_split=0.3)


model.save('CNN-CT')