import copy
import random

import numpy as np
import tensorflow as tf

from parameters import pms
from network.NetworkLinear import NetworkLinear
from network.NetworkNet import NetworkNet
from network.NetworkSVM import NetworkSVM
from network.svmlib.svmutil import *
from train.TrainBase import TrainBase
from environment import Environment
from recorder.RecorderBase import RecorderBase
from recorder.RecorderQFunc import RecorderQFunc
from AgentBase import AgentBase
class AgentSVM(AgentBase):
    def __init__(self):
        self.env = Environment(returnType=pms.return_type)
        obs_size = (pms.return_type + 1) * pms.system_state_num
        self.sess = None
        self.net = eval(pms.network_name)('%s_AgentBase'%pms.network_name,self.sess,obs_shape = obs_size)
        self.train = eval(pms.train_name)(self.net)
        self.train.sync_net()
        self.state_history = []
        self.action_history = []
        self.next_state_history = []
        self.reward_history = []

        self.recorder = eval(pms.recorder_name)()

    def save_model(self,model_name):
        for i in range(len(self.net.m_list)):
            svm_save_model('checkpoint/'+model_name+str(i),self.net.m_list[i])

    def load_model(self,model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.sess,model_name)
            else:
                self.saver.restore(self.sess,tf.train.latest_checkpoint(pms.checkpoint_dir))
        except:
            print "load fail!"