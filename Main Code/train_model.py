import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from keras import optimizers
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
import pandas as pd
import seaborn as sns

tf.executing_eagerly()

input_shape = (32,32)

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=0.1 ,width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_directory('C:\\Skripsi Aamiin\\data\\train', target_size=(input_shape), batch_size=1,class_mode='categorical')
validation_generator = train_datagen.flow_from_directory( 'C:\\Skripsi Aamiin\\data\\val',  target_size=(input_shape), batch_size=1,class_mode='categorical')

from keras.applications.vgg16 import VGG16 # Inisiasi untuk Arsitektur VGG (VGG16 atau VGG19)
from keras.models import Model 
#RUN FOR VGG
IMAGE_SIZE = [32,32] # Ukuran Gambar
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) 

#ADDING FOR CUSTOM LAYERS
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
prediction = Dense(36, activation='softmax')(x) # Inisiasi untuk jumlah maksimal dense 
model = Model(inputs=vgg.input, outputs=prediction)


model.compile(loss='categorical_crossentropy', # categorical_crosentropy untuk multilabe (lebih dari 2 kelas)
                    optimizer=tf.optimizers.Adam(0.001), # learning rate (estimation/itterate)
                    metrics=['accuracy'])
model.summary()

model_history=model.fit(train_generator,validation_data=validation_generator, epochs = 10)

model.save("C:\\Skripsi Aamiin\\cnn_ocr_data_vgg16.hdf5")

#loss Measurement
plt.plot(model_history.history['loss'], '--',label='train loss')
plt.plot(model_history.history['val_loss'], '' ,label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracy measurement
plt.plot(model_history.history['accuracy'], '--',label='train accuracy')
plt.plot(model_history.history['val_accuracy'], '',label='val accuracy')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')