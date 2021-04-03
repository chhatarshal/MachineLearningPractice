#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Creating Neural Network 

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# In[20]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[21]:


x_train.astype('float32')
y_train.astype('float32')
x_test.astype('float32')
y_test.astype('float32')


# In[22]:


x_test.shape


# In[23]:


x_train = x_train / 255
x_test = x_test / 255


# In[26]:


x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


# In[27]:


x_test.shape


# In[42]:


model = keras.models.Sequential(
[
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(248, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[43]:


model.compile()


# In[44]:


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)


# In[45]:


model.fit(x_train, y_train, epochs=10)


# In[ ]:




