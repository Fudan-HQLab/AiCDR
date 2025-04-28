import numpy as np
import keras
import tensorflow
import os
import sys
import multiprocessing as mp
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation,TimeDistributed,MaxPooling1D
from tensorflow.python.keras.layers import LSTM,GRU
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

aalist=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X","U","Z","B"," "]


def seqfrmat(seqinp,maxlnpep):
    tmp=seqinp.strip()
    while len(tmp)<=maxlnpep:
        tmp=tmp+" "
    coding=[]
    seqid=[]
    for x in range(0,maxlnpep+1):
        tmpctgr=to_categorical(aalist.index(tmp[x]), num_classes=len(aalist))
        coding.append(tmpctgr)
        seqid.append(aalist.index(tmp[x]))
    return seqid,coding

def loaddata(csvpath,csvpathneg,maxlnpep):
    f=open(csvpath,'r')
    ln=f.readlines()[:]
    lenln=len(ln)
    clnpep=[]
    clncoding=[]
    f.close()  
    fn=open(csvpathneg,'r')
    lnn=fn.readlines()[:]
    lenlnn=len(lnn)
    fn.close()
    datacutoff=0
    seqlist=sample(range(0,lenln),lenln)  
    seqlistneg=sample(range(0,lenlnn),lenlnn)
    for i in tqdm(range(0,lenln)):
        frmseq,frmcod=seqfrmat(ln[i],maxlnpep)
        frmcod=[[1]]
        clnpep.append(frmseq)
        clncoding.append(frmcod)
    for i in tqdm(range(0,lenlnn)):
        frmseq,frmcod=seqfrmat(lnn[i],maxlnpep)
        frmcod=[[0]]
        clnpep.append(frmseq)
        clncoding.append(frmcod)
    return clnpep,clncoding

def save_model(model):
    model_json = model.to_json()
    with open("Model-GRU256-512.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Model-GRU256-512.h5")
    print("Saved model to disk")

if __name__ == "__main__":
    maxlnpep=25
    nproc=4
    #Set the link to the positive and negative data
    Positive_set,Negative_set=loaddata("data_path/amp_all.csv","data_path/nonamp.csv",maxlnpep)

    X=np.array((Positive_set))
    Y=np.array((Negative_set))

    model = Sequential()
    aalstln=len(aalist)
    dataln=X.shape[1]
    #Model set-up
    model.add(Embedding(input_dim=aalstln, output_dim=len(aalist), input_length=dataln, mask_zero=False))
    model.add(GRU(units=256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(0.03)))
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=25))

    optimizer = Adam(lr=0.00001)
    model.build(input_shape=(aalstln,dataln))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history_callback = model.fit(X,Y,epochs=1000, batch_size=128,validation_split=0.2)

    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["accuracy"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_accuracy"]
    numpy_loss_history = np.array(loss_history)
    numpy_acc_history = np.array(acc_history)
    numpy_val_loss_history = np.array(val_loss_history)
    numpy_val_acc_history = np.array(val_acc_history)
    np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
    np.savetxt("acc_history.txt", numpy_acc_history, delimiter=",")
    np.savetxt("val_loss_history.txt", numpy_val_loss_history, delimiter=",")
    np.savetxt("val_acc_history.txt", numpy_val_acc_history, delimiter=",")
    save_model(model)


