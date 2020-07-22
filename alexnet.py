# # alexnet.py

# """ AlexNet.
# References:
#     - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
#     Classification with Deep Convolutional Neural Networks. NIPS, 2012.
# Links:
#     - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
# """

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# from tflearn.layers.normalization import local_response_normalization

# def alexnet(width, height, lr):
#     network = input_data(shape=[None, width, height, 1], name='input')
#     network = conv_2d(network, 96, 11, strides=4, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = conv_2d(network, 256, 5, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 384, 3, activation='relu')
#     network = conv_2d(network, 256, 3, activation='relu')
#     network = max_pool_2d(network, 3, strides=2)
#     network = local_response_normalization(network)
#     network = fully_connected(network, 4096, activation='tanh')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 4096, activation='tanh')
#     network = dropout(network, 0.5)
#     network = fully_connected(network, 3, activation='softmax')
#     network = regression(network, optimizer='momentum',
#                          loss='categorical_crossentropy',
#                          learning_rate=lr, name='targets')

#     model = tflearn.DNN(network, checkpoint_path='model_alexnet',
#                         max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

#     return model


# import argparse

# Import necessary components to build LeNet
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
# from keras.regularizers import l2

# def alexnet_model(img_shape=(150, 150, 1), n_classes=10, l2_reg=0.,weights=None):

#     # Initialize model
#     alexnet = Sequential()

#     # Layer 1
#     alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,padding='same', kernel_regularizer=l2(l2_reg)))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 2
#     alexnet.add(Conv2D(256, (5, 5), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 3
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(512, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 4
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(1024, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))

#     # Layer 5
#     alexnet.add(ZeroPadding2D((1, 1)))
#     alexnet.add(Conv2D(1024, (3, 3), padding='same'))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(MaxPooling2D(pool_size=(2, 2)))

#     # Layer 6
#     alexnet.add(Flatten())
#     alexnet.add(Dense(3072))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(Dropout(0.5))

#     # Layer 7
#     alexnet.add(Dense(4096))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('relu'))
#     alexnet.add(Dropout(0.5))

#     # Layer 8
#     alexnet.add(Dense(n_classes))
#     alexnet.add(BatchNormalization())
#     alexnet.add(Activation('softmax'))

#     if weights is not None:
#         alexnet.load_weights(weights)

#     return alexnet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import pickle
import datetime
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
def alexnet():
    width = 300
    height = 150
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(width, height, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis=1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis=1))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis=1))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # early_stop = EarlyStopping(patience=3, monitor='val_loss')
    adam = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    
    return model

