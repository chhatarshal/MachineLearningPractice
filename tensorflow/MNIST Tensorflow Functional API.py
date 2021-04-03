#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Creating Neural Network 

# In[48]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


# In[49]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[50]:


x_train.astype('float32')
y_train.astype('float32')
x_test.astype('float32')
y_test.astype('float32')


# In[51]:


x_test.shape


# In[52]:


x_train = x_train / 255
x_test = x_test / 255


# In[53]:


x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


# In[54]:


x_test.shape


# In[55]:


inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# In[56]:


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)


# In[57]:


model.fit(x_train, y_train, epochs=10)


# In[46]:





# In[ ]:




