#!/usr/bin/env python
# coding: utf-8

# In[42]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy


# In[43]:


data = keras.datasets.fashion_mnist.load_data() 


# In[58]:


(x_train, y_train), (x_test, y_test) = data


# In[59]:


x_train.shape


# In[60]:


y_train.shape


# In[61]:


len(numpy.unique(y_test))


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


x_train[1, :].shape


# In[ ]:





# In[64]:


plt.imshow(x_train[100,:])


# In[65]:


x_train.astype('float32') 
x_test.astype('float32')


# In[66]:


model = keras.models.Sequential(
[
    keras.layers.Input(shape=(28*28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[67]:


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)


# In[68]:


x_train = x_train.reshape(-1, 28*28)


# In[69]:


model.fit(x_train, y_train, epochs=4)


# In[71]:


model.fit(x_train, y_train, epochs=6)


# In[ ]:




