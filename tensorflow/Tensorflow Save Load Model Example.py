#!/usr/bin/env python
# coding: utf-8

# In[85]:


import tensorflow as tf
import numpy as np


# In[86]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[87]:


input_1 = tf.keras.layers.Input(shape=(28*28))


# In[88]:


layer1 = tf.keras.layers.Dense(128, activation='relu')(input_1)


# In[89]:


output_1 = tf.keras.layers.Dense(10, activation='softmax')(layer1)


# In[90]:


model = tf.keras.Model(inputs= input_1, outputs = output_1)


# In[91]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[92]:


x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


# In[93]:


model.compile(
loss= tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=['accuracy']
)


# # We will evaluate model without training

# In[96]:


model.evaluate(x_test, y_test)


# We saw that accuracy is very low 0.1057. Now we will train model

# In[95]:


model.fit(x_train, y_train, epochs=10)


# Now we have achived a very good accuracy we will save model

# In[97]:


model.save('saved_Model/')


# # We will create model again and will try to load model saved earlier

# In[70]:


input_1 = tf.keras.layers.Input(shape=(28*28))


# In[71]:


layer1 = tf.keras.layers.Dense(128, activation='relu')(input_1)


# In[72]:


output_1 = tf.keras.layers.Dense(10, activation='softmax')(layer1)


# In[73]:


model = tf.keras.Model(inputs= input_1, outputs = output_1)


# In[74]:


model.compile(
loss= tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=['accuracy']
)


# We create model now we will try to evaluate without training / loading model

# In[78]:


model.evaluate(x_test, y_test)


# We saw model gives very low accuracy 0.1320. SO this time we will load earlier saved model

# In[99]:


model3 = tf.keras.models.load_model("saved_Model/")


# Lets evaluate loaded model.

# In[100]:


model3.evaluate(x_test, y_test)


# Still very low accuracy we need to compile model before we evaluate it.

# In[103]:


model3.compile(
loss= tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=['accuracy']
)


# In[104]:


model3.evaluate(x_test, y_test)


# Now we get good accuracy. He we learn how to save and load model
