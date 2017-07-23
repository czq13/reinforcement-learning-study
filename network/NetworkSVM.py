import random
import copy

from svmlib.svmutil import *
import numpy as np

from NetworkBase import NetworkBase
from parameters import pms

class NetworkSVM(NetworkBase):
    def __init__(self,scope,sess,obs_shape = pms.system_state_num,action_shape = pms.system_action_num):
        self.x = np.random.rand(action_shape,100,obs_shape)
        self.y = np.random.rand(100,action_shape)

        self.param = svm_parameter('-s 4 -t 2 -c 4')
        self.action_shape = action_shape
        self.data_len = 100
        self.sess = None
        self._train()

    def train(self, state_set, target_set):
        for i in range(self.action_shape):
            pos = np.abs(target_set[:,i]) > 0
            tmp_x = state_set[pos,:]
            tmp_y = target_set[pos,:]
            predic_y,_,_ = svm_predict(tmp_y[:,i].tolist(),tmp_x.tolist(),self.m_list[i])
            pos = np.abs(predic_y - tmp_y[:,i]) > 0
            tx = np.concatenate([tmp_x[pos,:],self.x[i,:,:]],axis=0)
            ty = np.concatenate([tmp_y[:,i],self.y[:,i]],axis=0)
            if tx.shape[0] > self.data_len:
                self.x[i,:,:] = tx[0:self.data_len,:]
                self.y[:,i] = ty[0:self.data_len]
        self._train()

    def _train(self):
        self.m_list = []
        for i in range(self.action_shape):
            ty = self.y[:,i]
            prob = svm_problem(ty.tolist(),self.x[i,:,:].tolist())
            m = svm_train(prob, self.param)
            self.m_list.append(m)

    def get_weight(self):
        return [self.y,self.x]

    def set_wight(self,var_list):
        self.y = var_list[0]
        self.x = var_list[1]
        self._train()

    def get_output(self,obs):
        len = obs.shape[0]
        ans = np.zeros([len,self.action_shape])
        for i in range(self.action_shape):
            tmp,_,_ = svm_predict(range(len),obs.tolist(),self.m_list[i])
            ans[:,i] = np.array(tmp)
        return ans
