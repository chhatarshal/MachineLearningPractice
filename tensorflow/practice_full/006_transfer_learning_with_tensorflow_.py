# -*- coding: utf-8 -*-
"""006 Transfer Learning With Tensorflow .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19TLiBGtMWf6NX3n9MKp9Jm0MyFf7pWcv
"""

import tensorflow as tf

!wget https://raw.githubusercontent.com/chhatarshal/MachineLearningPractice/e878dbd141610a82206b7389c299c41ffe3765d3/tensorflow/helper/helper_functions.py

from helper_functions import unzip_data

# Get 10% of training data of 10 classes of Food101
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip
unzip_data('10_food_classes_10_percent.zip')

train_dir = '10_food_classes_10_percent/train'
test_dir = '10_food_classes_10_percent/test'

IMG_SIZE = (224, 224)
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir, label_mode="categorical", image_size=IMG_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir, label_mode="categorical", image_size=IMG_SIZE, shuffle=False) # don't s

check_point_path = '10_class_10_ptc_checkpoint'

check_point_dir = tf.keras.callbacks.ModelCheckpoint(check_point_path, monitor="val_accuracy", save_best_only=True,
                                                     save_weights_only=True)

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

data_augmentation = Sequential([
      preprocessing.RandomFlip("horizontal"),
      preprocessing.RandomRotation(0.2),
      preprocessing.RandomHeight(0.2),
      preprocessing.RandomZoom(0.2)                          
])

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))

x = data_augmentation(input_layer)

x = base_model(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

modle = tf.keras.Model(input_layer, outputs)

modle.summary()

# Compile
modle.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])

# Fit
history_all_classes_10_percent = modle.fit(train_data,
                                           epochs=1, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir]) # save best model weights to file

history_all_classes_10_percent = modle.fit(train_data,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir]) # save best model weights to file

history_all_classes_10_percent = modle.fit(train_data,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir]) # save best model weights to file

for layer_number, layer in enumerate(modle.layers):
  print(layer_number, layer.name)

#base_model
for layer_number, layer in enumerate(base_model.layers):
  if layer_number > 100:
    print(layer_number, layer.name)
    layer.trainable = True

history_all_classes_10_percent = modle.fit(train_data,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir]) # save best model weights to file

modle.summary()

# Compile
modle.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])

modle.summary()

URL = 'https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip'

!wget https://raw.githubusercontent.com/chhatarshal/MachineLearningPractice/e878dbd141610a82206b7389c299c41ffe3765d3/tensorflow/helper/helper_functions.py

# Get 10% of training data of 10 classes of Food101
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip

from helper_functions import unzip_data
unzip_data('101_food_classes_10_percent.zip')

train_dir1 = '101_food_classes_10_percent/train'
test_dir1 = '101_food_classes_10_percent/test'

import tensorflow as tf
IMG_SIZE = (224, 224)
train_data1 = tf.keras.preprocessing.image_dataset_from_directory(train_dir1, label_mode="categorical", image_size=IMG_SIZE)
test_data1 = tf.keras.preprocessing.image_dataset_from_directory(test_dir1, label_mode="categorical", image_size=IMG_SIZE)
input_layer_model_2 = tf.keras.Input(shape=(224, 224, 3), name="input_layer_model2")
base_model_2 = tf.keras.applications.EfficientNetB0(include_top=False)
base_model_2.trainable = True

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

data_augmentation = Sequential([
      preprocessing.RandomFlip("horizontal"),
      preprocessing.RandomRotation(0.2),
      preprocessing.RandomHeight(0.2),
      preprocessing.RandomZoom(0.2)                          
])



x = data_augmentation(input_layer_model_2)
x = data_augmentation(x)
x = base_model_2(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(101, activation="softmax")(x)

model_2 = tf.keras.Model(input_layer_model_2, outputs)

check_point_path2 = '101_class_10_ptc_checkpoint'

check_point_dir2 = tf.keras.callbacks.ModelCheckpoint(check_point_path2, monitor="val_accuracy", save_best_only=True,
                                                     save_weights_only=True)

# Compile
model_2.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])

# Fit
history_all_classes_101_percent = model_2.fit(train_data1,
                                           epochs=1, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data1,
                                           validation_steps=int(0.15 * len(test_data1)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir2]) # save best model weights to file

history_all_classes_101_percent = model_2.fit(train_data1,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data1,
                                           validation_steps=int(0.15 * len(test_data1)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir2]) # save best model weights to file

from helper_functions import plot_loss_curves

plot_loss_curves(history_all_classes_101_percent)

for layer_number, layer in enumerate(base_model_2.layers):
  if layer_number < 200:
    layer.trainable = False 
  #print(layer_number, layer.name)

import tensorflow as tf
IMG_SIZE = (224, 224)
train_data1 = tf.keras.preprocessing.image_dataset_from_directory(train_dir1, label_mode="categorical", image_size=IMG_SIZE)
test_data1 = tf.keras.preprocessing.image_dataset_from_directory(test_dir1, label_mode="categorical", image_size=IMG_SIZE)
input_layer_model_2 = tf.keras.Input(shape=(224, 224, 3), name="input_layer_model2")
base_model_2 = tf.keras.applications.EfficientNetB0(include_top=False)
base_model_2.trainable = True

x = data_augmentation(input_layer_model_2)
x = data_augmentation(x)
x = base_model_2(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(101, activation="softmax")(x)

for layer_number, layer in enumerate(base_model_2.layers):
  if layer_number < 200:
    layer.trainable = False 
  #print(layer_number, layer.name)

# Compile
model_2.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])

# Fit
history_all_classes_101_percent = model_2.fit(train_data1,
                                           epochs=1, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data1,
                                           validation_steps=int(0.15 * len(test_data1)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir2]) # save best model weights to file

thistory_all_classes_101_percent = model_2.fit(train_data1,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data1,
                                           validation_steps=int(0.15 * len(test_data1)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir2])

thistory_all_classes_101_percent = model_2.fit(train_data1,
                                           epochs=10, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data1,
                                           validation_steps=int(0.15 * len(test_data1)), # evaluate on smaller portion of test data
                                           callbacks=[check_point_dir2])

plot_loss_curves(thistory_all_classes_101_percent)

