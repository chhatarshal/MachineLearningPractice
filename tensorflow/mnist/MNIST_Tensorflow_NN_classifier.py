#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[ ]:


# Import for Operating system operation
import os
# Import tensorflow
import tensorflow as tf
#import keras API tensorflow
from tensorflow import keras
#import layers from keras api
from tensorflow.keras import layers
# import mnist from keras datasets
from tensorflow.keras.datasets import mnist


# In[17]:


# load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape tensors to flat and divide by 255 for making traning stable
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
# do same on test data
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0


# One simple way to create model is using keras sequential api 

# In[18]:


model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)


# Another way to create same

# In[21]:


model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu", name="my_layer"))
model.add(layers.Dense(10))


# Below is function API

# In[22]:


inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[14]:


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)


# In[15]:


model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# In[ ]:





# In[ ]:




