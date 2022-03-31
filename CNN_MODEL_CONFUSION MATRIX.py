#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:58:38 2022

@author: lukas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D 
import numpy as np

import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)




train_path='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_VOLUMEPURE/TRAIN'
valid_path='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_VOLUMEPURE/VALID'
test_path='/Users/lukas/Documents/FICHIER_STRATGIE_TRADING_VOLUMEPURE/TEST'



def plotImages(images_arr):
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        
train_batches=ImageDataGenerator().flow_from_directory(directory=train_path,target_size=(500,700), classes=['BAISSIER','HAUSSIER'],batch_size=10)
valid_batches=ImageDataGenerator().flow_from_directory(directory=valid_path,target_size=(500,700), classes=['BAISSIER','HAUSSIER'],batch_size=10)
test_batches=ImageDataGenerator().flow_from_directory(directory=test_path,target_size=(500,700), classes=['BAISSIER','HAUSSIER'],batch_size=10)


imgs, labels = next(train_batches)


plotImages(imgs)
print(labels)


model=Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(500,700,3)),
    MaxPool2D(pool_size=(2,2),strides=2),
    Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=2),
    Flatten(),
    Dense(units=2,activation='softmax'),
    ])


model.summary()


model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches,epochs=30,verbose=1)


test_imgs,test_labels=next(test_batches)
plotImages(test_imgs)
print(test_labels)
test_batches.classes



prediction=model.predict(x=test_batches,verbose=0)


np.round(prediction)

cm=confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(prediction,axis=-1))

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)    
    plt.title(title)
    plt.colorbar()
    ticks_marks=np.arange(len(classes))
    plt.xticks(ticks_marks,classes,rotation=45)
    plt.yticks(ticks_marks,classes)
    
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix,without normalization')
    print(cm)
     
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
            horizontalalignment="center",                                 
            color="white" if cm[i,j]>thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

test_batches.class_indices

cm_plot_labels=['Baissier','Haussier']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')



model.save('CNN MODEL WITH CONFUSION MATRIX')


