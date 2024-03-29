# -*- coding: utf-8 -*-
"""NthDigitProblem.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H8goN27T8RoqlVdSuuFKwD-YPyfT0CII
"""

def findNthDigit(n):
  digit = 1
  digits_interval =9  
  while n - digits_interval > 0:
    n = n - digits_interval
    digit = digit + 1
    digits_interval = 10**(digit - 1)*9*digit
    if digits_interval < 0:
      break  
  base = 10**(digit-1)
  number = base + int((n - 1) / digit)   

  return str(number)[((n - 1) % digit)]
  #return number

def doubler(n):
  return n*2

findNthDigit(83)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
                             tf.keras.layers.Dense(1),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(512, activation='relu'),
                             tf.keras.layers.Dense(1)
])

model.compile(
    loss = tf.keras.losses.mae,
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["mae"]
)

model.fit([10], [2], epochs=1)

import numpy as np
inlot = []
outlot = []
for i in range(100000):
  #print(np.random.randint(10000))
  n = np.random.randint(10000)
  inlot.append(n)
  #r = int(doubler(n))
  r = int(findNthDigit(n))
  outlot.append(r)
  #print(i % 10)
 # if i % 10 is 0:
    #print(inlot)
   # print(outlot)

len(inlot)

model.fit(inlot, outlot, epochs=10, batch_size=20)

model.predict([555])

findNthDigit(555)

