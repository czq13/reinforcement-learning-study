from abc import ABCMeta,abstractmethod

import tensorflow as tf

class NetworkBase():
    __metaclass__ = ABCMeta

    def __init__(self,scope):
        pass

    @abstractmethod
    def train(self,state_set,target_set):
        pass

    def get_weight(self):
        return [v.eval(self.sess) for v in tf.trainable_variables()]

    def set_wight(self,var_list):
        for (v,tv) in zip(var_list,self.var_list):
            self.sess.run(tf.assign(tv,v))

    def get_output(self,obs):
        return self.sess.run(self.output,{self.obs:obs})
