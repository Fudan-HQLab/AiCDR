import keras
import tensorflow as tf
import os
import sys
import multiprocessing as mp
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense,MaxPooling1D, GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.layers import Dropout
import numpy as np
import random
import sys
from keras.utils.np_utils import to_categorical
from tqdm import *
from keras.preprocessing import sequence
from keras.models import model_from_json
from random import sample
from keras.callbacks import CSVLogger



#Defining the vocabulary (should be constant with the vocabulary used for the generative model)
#aalist=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"," "]

def seq_to_one_hot(sequence):
    coded_seq = []
    for letter in sequence:
        encoding = [0] * len(aalist)
        encoding[aalist.index(letter)] = 1
        coded_seq.append(encoding)
    padded_seq = coded_seq + [[0] * len(aalist)] * (maxlnpep - len(sequence))
    return padded_seq

def loaddata(positive_csv, negative_csv, maxlnpep):
    X = []
    y = []

    with open(positive_csv, 'r') as file:
        for line in file:
            sequence = line.strip()
            encoded_sequence = seq_to_one_hot(sequence)
            X.append(encoded_sequence)
            y.append(1)  # Positive label

    with open(negative_csv, 'r') as file:
        for line in file:
            sequence = line.strip()
            encoded_sequence = seq_to_one_hot(sequence)
            X.append(encoded_sequence)
            y.append(0)  # Negative label

    X = np.array(X)
    y = np.array(y)
    return X, y

def create_model(maxlnpep, aalist):
    input_layer = Input(shape=(maxlnpep, len(aalist)))

    transformer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
    attention_output = transformer(input_layer, input_layer)

    dense1 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(attention_output)
    dropout1 = Dropout(0.5)(dense1)
    dense3 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout1)
    dropout4 = Dropout(0.5)(dense3)

    dense5 = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout4)
    model = Model(inputs=input_layer, outputs=dense5)

    optimizer = Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model


def save_model(model):
    model_json = model.to_json()
    with open("Model-GRU256-512.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Model-GRU256-512.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    maxlnpep=25
    nproc=4
    aalist=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"]
    #Set the link to the positive and negative data
    X, y=loaddata("data_path/amp_all.csv","data_path/nonamp.csv",maxlnpep)



    model = create_model(maxlnpep, aalist)
    history_callback = model.fit(X, y, epochs=5000, batch_size=512, validation_split=0.2)
    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["accuracy"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_accuracy"]

    np.savetxt("loss_history.txt", np.array(loss_history), delimiter=",")
    np.savetxt("acc_history.txt", np.array(acc_history), delimiter=",")
    np.savetxt("val_loss_history.txt", np.array(val_loss_history), delimiter=",")
    np.savetxt("val_acc_history.txt", np.array(val_acc_history), delimiter=",")
    save_model(model) 



