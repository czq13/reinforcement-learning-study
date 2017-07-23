import copy
import random

import numpy as np
import tensorflow as tf

from parameters import pms
from network.NetworkLinear import NetworkLinear
from network.NetworkNet import NetworkNet
from train.TrainBase import TrainBase
from environment import Environment
from recorder.RecorderBase import RecorderBase
from recorder.RecorderQFunc import RecorderQFunc
class AgentBase():
    def __init__(self):
        self.env = Environment(returnType=pms.return_type)
        obs_size = (pms.return_type + 1) * pms.system_state_num
        self.sess = tf.Session()
        self.net = eval(pms.network_name)('%s_AgentBase'%pms.network_name,self.sess,obs_shape = obs_size)
        self.train = eval(pms.train_name)(self.net)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=50)
        self.train.sync_net()
        self.state_history = []
        self.action_history = []
        self.next_state_history = []
        self.reward_history = []

        self.recorder = eval(pms.recorder_name)()

    def sample(self,test = False):
        self.state_history = []
        self.action_history = []
        self.u_history = []
        self.next_state_history = []
        self.reward_history = []
        self.q = []
        state = self.env.reset()
        for i in range(pms.maxlen):
            self.state_history.append(copy.deepcopy(state))
            u_q = self.net.get_output(state)
            self.q.append(u_q.copy())
            tr = random.random()
            if (tr < pms.epsilon and not test):
                u = np.random.randint(0,pms.system_action_num)
            else:
                u = np.argmax(u_q)
            if not pms.discrete:
                action = self.env.action_space[u]
            else:
                action = u
            self.u_history.append(action)
            ru = np.zeros([pms.system_action_num,])
            ru[u] = 1
            state, reward, done, info = self.env.step(action)
            self.next_state_history.append(copy.deepcopy(state))
            self.reward_history.append(np.array([[reward]]))
            self.action_history.append(np.array([ru]))
            if done:
                break

    def train_net(self):
        for i in range(pms.epoch_num):
            self.sample()
            self.summary()
            self.train.add_history(
                np.concatenate(self.state_history,axis = 0),
                np.concatenate(self.action_history,axis = 0),
                np.concatenate(self.next_state_history,axis = 0),
                np.concatenate(self.reward_history,axis = 0)
            )
            self.train.train_epoch()
            if i % pms.sync_cycle:
                self.train.sync_net()
            if i % pms.test_frequence == 0:
                self.sample(test = True)
                self.summary(test = True)
            if i % pms.save_model_fre == 0:
                self.save_model(pms.environment_name + '_' + str(i / pms.save_model_fre))

    def summary(self,test = False):
        q_his = np.concatenate(self.q,axis = 0)
        if test:
            self.test_avg_reward = np.average(self.reward_history)
            self.test_total_reward = np.sum(self.reward_history)
            self.recorder.add_data(['test_avg_reward','test_total_reward'],[self.test_avg_reward,self.test_total_reward])
            self.recorder.set_data(['test_action','test_q0','test_q1'],[self.u_history,q_his[:,0],q_his[:,1]])
        else:
            self.train_avg_reward = np.average(self.reward_history)
            self.train_total_reward = np.sum(self.reward_history)
            self.recorder.add_data(['train_avg_reward','train_total_reward'],[self.train_avg_reward,self.train_total_reward])
            self.recorder.set_data(['train_action','train_q0','train_q1'],[self.u_history,q_his[:,0],q_his[:,1]])

    def save_model(self,model_name):
        self.saver.save(self.sess,'checkpoint/'+model_name+'.ckpt')

    def load_model(self,model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.sess,model_name)
            else:
                self.saver.restore(self.sess,tf.train.latest_checkpoint(pms.checkpoint_dir))
        except:
            print "load fail!"
