from abc import ABCMeta,abstractmethod

import tensorflow as tf
import numpy as np

from parameters import pms
from network.NetworkLinear import NetworkLinear
from network.NetworkNet import NetworkNet
from network.NetworkSVM import NetworkSVM

class TrainBase():
    def __init__(self,net):
        self.history_len = 0
        self.state_history = None
        self.action_history = None
        self.next_state_history = None
        self.reward_history = None
        obs_size = (pms.return_type + 1) * pms.system_state_num
        self.net = eval(pms.network_name)(('%s_TrainBase')%pms.network_name,net.sess,obs_shape = obs_size)
        self.net_sync = net

    def add_history(self,state_his,action_his,next_state_history,reward_his):
        if self.history_len == 0:
            self.state_history = state_his
            self.action_history = action_his
            self.next_state_history = next_state_history
            self.reward_history = reward_his
        else:
            self.state_history = np.concatenate([state_his,self.state_history],axis=0)
            self.action_history = np.concatenate([action_his,self.action_history],axis=0)
            self.next_state_history = np.concatenate([next_state_history,self.next_state_history],axis=0)
            self.reward_history = np.concatenate([reward_his,self.reward_history],axis=0)
        self.history_len += len(state_his)
        if self.history_len > pms.history_len:
            self.state_history = self.state_history[0:pms.history_len,:]
            self.action_history = self.action_history[0:pms.history_len,:]
            self.next_state_history = self.next_state_history[0:pms.history_len,:]
            self.reward_history = self.reward_history[0:pms.history_len,:]
            self.history_len = pms.history_len

    def replay_history(self):
        list = np.random.randint(0,self.history_len,pms.batch_size)
        return self.state_history[list,:],self.action_history[list,:],self.next_state_history[list,:],\
               self.reward_history[list,:]

    def train_epoch(self):
        for i in range(self.history_len/pms.batch_size):
            shis,ahis,nhis,rhis = self.replay_history()
            q = self.net.get_output(shis) * ahis
            max_next_q = np.max(self.net.get_output(nhis),axis = 1)
            ttarget_q = (1-pms.gamma) * (rhis[:,0] + max_next_q)
            action_size = ahis.shape[1]
            target_q = np.tile(ttarget_q[:,np.newaxis],[1,action_size]) * ahis + q * pms.gamma
            self.net_sync.train(shis,target_q)

    def sync_net(self):
        var_list = self.net_sync.get_weight()
        self.net.set_wight(var_list)

