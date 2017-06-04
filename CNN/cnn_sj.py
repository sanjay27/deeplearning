#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:45:46 2017
Modified: 04Jun 2017

@author: sanjayjadhav
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

## Part 0 - Preprocesing is done with Folder structure, so NO Part 1 for Data Prepocessing as we did eaerlier

## Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D          # first step = Convolutional layers (since imgs are 2D we use Conv2D)
from keras.layers import MaxPooling2D    # second step = Pooling ()
from keras.layers import Flatten         # third step  = Flatten into large feature vector
from keras.layers import Dense           # 4th   step  = Add fully connected layers

# Initialising the CNN   - for sequence of layers
classifier = Sequential()

# Adding Layers 
# Step 1 - Convolution      #(first layer of CNN)  feature detectors applied to get feature maps
#  classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))
classifier.add(Conv2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))
                        # start with 32 filters (i.e. feature detectors) then 128, 256 with 3 x 3 dimensions, 
                        # input shape is very important to force them into fixed size  RGB then 3 channels, B&W 2 channels
                        # in TF backend use shape first then channels (64,64,3)
                        # to get non-negative values in our CNN and classifier is non-linear function, we need  to use  non-linearity and we use activation function 

# Step 2 - Pooling          #reducing the size of the feature map 2x2 sub-table and take max of 4 cells and therefore to reduce the number of the nodes in fully connected layers. With the stride of 2 (moving 2 blocks right)
classifier.add(MaxPooling2D(pool_size = (2,2)))   
                                                  # reduced the complexity of model without reducing its performance. 2x2 size in general - we still keep the information and be precise. 
                                                  # And it will Divide by 2.
                                                  # becoz we want to reduce the number of Vectors & Nodes for the next step.  to make it less complex and reduce computation.

# Step 3 - Flattening       # output of feature maps as single Vector with pixel's spatial structure to future connected layers
classifier.add(Flatten())   # no need to pass any arguments here, keras knows to take the object defined previously
                            # some parethesis becoz its a function


# Step 4 - Full connection  # final step, make Hidden layer i.e. Fully Connected Layer making Classing ANN of fully connected layers
classifier.add(Dense(units = 128, activation = 'relu'))   # input layer, common practise is choose no. between Input and output numbers. Not too small nor too large and power of 2 will be good.
classifier.add(Dense(units = 1, activation = 'sigmoid'))   # output layer  - sigmoid funcion becoz binary outcome, 
                                                           # if >2 categories for classification then softmax

# Now we added all the Layers and now we have build the Full Convolutional Network with 4 steps see screenshot "building cnn layers viz.png"


## Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy' ])
                                # stochastic gradient, 
                                #loss function becoz logrithmic loss & binary outcome (if cross_entropy if >2), 
                                #performance metric)

## Part 2 - Fitting the CNN to the images  - Image Prepocessing Step to prevent overfitting.
   # image augmetation  - enriching our datasets (traning sets)without any more images and therefore good performance and less overfitting.
   # keras documenation: https://keras.io/preprocessing/image/
   # We either need more Images or have to use Image Preprocessing!!!  rotating them, etc.. hence more images to train - hence trng imgs are augmented.
from keras.preprocessing.image import ImageDataGenerator
                                        # Using .flow_from_directory(directory) function instead of flow
train_datagen = ImageDataGenerator(     #this object will be used to augment the images of trng set
        rescale=1./255,                 #scale the images between 0 and 1 becoz 1/255
        shear_range=0.2,                #random transformation
        zoom_range=0.2,                 #random zooms transformation
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)   #same to augment the the iamges of Test set.  rescale the images of Test Set

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set', # directory of training images
                                                target_size=(64, 64),   # becoz above 64 x 64 in convolutional step using Conv2D
                                                batch_size=32,          # batchsize, after which the weights will be updated
                                                class_mode='binary')    # DV is binary or >2 Cats.
                                                                        # outcome =  Found 2000 images belonging to 2 classes. bocoz of folders arranged.
test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),  # images of test_set will expect 64x64 by our CNN
                                            batch_size=32,
                                            class_mode='binary')
                                                                    # outcome =  Found 2000 images belonging to 2 classes. bocoz of folders arranged.

classifier.fit_generator(training_set,             # model fit generator - apply on our model called classifier
                        steps_per_epoch=8000,      # number of images in our training set!!! all observations of trng set pas thru CNN during each epoch, therefore 8K
                        epochs=25,
                        validation_data=test_set,  # test set on which we want to evaluate our performance!!!
                        validation_steps=2000)     # num of images in our test set
                         

