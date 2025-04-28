from __future__ import print_function
from subprocess import Popen, PIPE
from math import *
import random,os
import random as pr
from copy import deepcopy
import itertools
import time
import math
import argparse
import subprocess
# from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import pad_sequences
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from mpi4py import MPI
import sys,os
from multiprocessing import Pool
from random import sample
import numpy as np


def redistribution(idx, total):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l))
            #print(rescalar[s])
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
            #print(ret[i, j])
    return ret

# classification subroutine
def critic1(criticmod, intseq):

    aalist=['a','r','n','d','c','q','e','g','h','i','l','k','m','f','p','s','t','w','y','v','x']
    aalen=len(aalist)
    cri_seq_out=[]
    for seq in intseq:
        # print(f'seq: {seq}')
        encoding = np.zeros((len(seq), aalen))
        for i, letter in enumerate(seq):
            if letter in aalist:
                encoding[i][aalist.index(letter)] = 1
        encoding = np.expand_dims(encoding, axis=0)
        predictions=criticmod.predict(encoding)
        cri_seq_out.append(predictions[0][0][0])

    # for i in range(0,len(intseq)):
    #     encoding = np.eye(len(aalist))[intseq[i]]
    #     # encoding[aalist.index(letter)] = 1
    #     # coded_seq.append(encoding)
    #     encoding = np.expand_dims(encoding, axis=1)
    #     predictions=criticmod.predict(encoding)
    #     cri_seq_out.append(predictions[0][0])

    return cri_seq_out

def critic2(criticmod, intseq):
    aalist=['a','r','n','d','c','q','e','g','h','i','l','k','m','f','p','s','t','w','y','v','x','u','z','b',' ']
    aalen=len(aalist)
    cri_seq_out=[]
    for i in range(0,len(intseq)):
        #print(f'1:{1}')
        #print(f'intseq[i]: {intseq[i]}')
        x=np.reshape(intseq[i],(1,len(intseq[i])))
        x_pad= pad_sequences(x, maxlen=aalen+1, dtype='int32',padding='post', truncating='pre', value=25)
        #print(f'x.shape and aalen and x_pad: {x.shape}, {aalen}, {x_pad.shape}')
        predictions=criticmod.predict(x_pad)
        cri_seq_out.append(predictions[0][0][0])

    return cri_seq_out

def loadModel(path, filename):
    json_file = open(path+"/"+filename+".json","r")
    RNNjson = json_file.read()
    json_file.close()
    loadmodel = model_from_json(RNNjson)
    loadmodel.load_weights(path+"/"+filename+".h5")
    return loadmodel


class Reward(object):
    def __init__(self, model, dis, sess, rollout_num):
        self.model = model
        self.dis = dis
        self.sess = sess
        self.rollout_num = rollout_num
    

    def get_reward(self, input_x):
        rewards = []
        lambda_d = 0.6
        lambda_g1 = 0.3
        lambda_g2 = 0.1
        
        criticmod1 = loadModel("../../Guider1", "Model-transformer-512")
        criticmod2 = loadModel("../../Guider2", "Model-GRU256-512")
        print(f'self.model.sequence_length // self.model.step_size: {self.model.sequence_length // self.model.step_size}')
        for i in range(self.rollout_num):
            for given_num in range(1, self.model.sequence_length // self.model.step_size):
                real_given_num = given_num * self.model.step_size
                # print(f'real_given_num: {real_given_num}')
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.drop_out: 1.0}
                samples = self.sess.run(self.model.gen_for_reward, feed)

                cri_seq1=critic1(criticmod1, samples)
                #print(f'cri_seq1: {cri_seq1}')

                # print(samples)

                cri_seq2=critic2(criticmod2, samples)
                #print(f'cri_seq2: {cri_seq2}')

                feed = {self.dis.D_input_x: samples}
                ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                # print(f'ypred: {ypred}')
                ypred = lambda_d*np.array(ypred)+lambda_g1*np.array(cri_seq1) + lambda_g2*np.array(cri_seq2)
                #print(f'ypred : {ypred}  {ypred.shape}')
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred
            cri_seq1=critic1(criticmod1, input_x)
            cri_seq2=critic2(criticmod2, input_x)
            feed = {self.dis.D_input_x: input_x}
            ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            ypred = lambda_d*np.array(ypred)+lambda_g1*np.array(cri_seq1) + lambda_g2*np.array(cri_seq2)
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)
        
        return rewards






