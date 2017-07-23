import copy

import gym
import numpy as np

from parameters import pms

class Environment():
    def __init__(self,returnType = 0):
        self._env = gym.make(pms.environment_name)
        self.returnType = returnType
        if (returnType == 0):
            self.stateNum = pms.system_state_num
        elif (returnType == 1):
            self.stateNum = pms.system_state_num * 2

        self.state = np.zeros([self.stateNum,],dtype=float)

        self.obs = self.state.copy()
        self.obs_limit = np.zeros_like(self.state,dtype=float) + 0.0001
        self.update_limit()
        #if pms.discrete == False
        self.action_space = None

    def step(self,action):
        tstate,reward,done,info = self._env.step(action)
        self._reset_state(tstate)
        self.get_obs()
        if pms.environment_name == 'CartPole-v0':
            if done:
                reward = -1
            else:
                reward = 0
        return copy.deepcopy(self.obs.reshape([1,self.stateNum])),reward,done,info

    def get_obs(self):
        self.obs = self.state.copy()
        if pms.update_limit:
            self.update_limit()
            self.obs = self.obs / self.obs_limit

    def _reset_state(self,tstate):
        if (self.returnType == 0):
            self.state = tstate
        elif (self.returnType == 1):
            self.state = np.concatenate([tstate,tstate*tstate],axis = 0)

    def update_limit(self):
        self.obs_limit = np.max(np.abs(np.concatenate([self.obs_limit[:,np.newaxis],self.obs[:,np.newaxis]],axis=1)),axis=1)

    def reset(self):
        tstate = self._env.reset()
        self._reset_state(tstate)
        return copy.deepcopy(self.state.reshape([1,self.stateNum]))
