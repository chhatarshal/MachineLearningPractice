import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

import requests
@st.cache
def download(url, name):
    r = requests.get(url, allow_redirects=True)
    open(name, 'wb').write(r.content)

url = 'https://raw.githubusercontent.com/chhatarshal/MachineLearningPractice/main/tensorflow/helper/helper_functions.py'
download(url, 'helper_functions.py')
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys


url2 = 'https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip'
download(url2, 'nlp_getting_started.zip')
# Unzip data
unzip_data("nlp_getting_started.zip")


st.write("Reading data:")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
st.write(train_df.head())

# Shuffle training dataframe
# Shuffle training dataframe
st.write("Lets a shuffle data and take a sample and try to see what is inside:")
train_df_shuffled = train_df.sample(frac=1, random_state=42) 
train_df_shuffled.head()
st.write(train_df_shuffled.head())

st.write("Lets explore some test data: ")

st.write(test_df.head())

st.write('Lets see how many positive and negative sample we have')

st.write(train_df.target.value_counts())

st.write('Here we see we have 4342 records says tweet are not disaster and rest 3271 tweets are disaster')

st.write(len(train_df), len(test_df))

st.write('We have 7613 train data and 3263 test data')
st.write('Lets now visualize some random data')

import random

for i in range(10):
  random_index = random.randint(0, len(train_df))
  train_df.iloc[random_index]["text"]
  text, target = train_df.iloc[random_index][["text", "target"]]
  st.write(text, '\n', target)
  st.write(f"\n\n")


st.write('We have to divide data between training and validation data and sklearn is best tool to do that')


from sklearn.model_selection import train_test_split

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, # use 10% of training data for validation split
                                                                            random_state=42)
st.write(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))

st.write(train_sentences[:10], train_labels[:10])