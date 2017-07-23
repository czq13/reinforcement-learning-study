
import tensorflow as tf
from NetworkBase import NetworkBase
from parameters import pms

class NetworkLinear(NetworkBase):
    def __init__(self,scope,sess,obs_shape = pms.system_state_num,action_shape = pms.system_action_num):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = tf.placeholder(
                tf.float32,shape=[None,obs_shape],name="%s_obs"%scope
            )
            self.w = tf.get_variable('Matrix',[obs_shape,action_shape],tf.float32,
                                     tf.random_normal_initializer(stddev=0.02))
            self.output = tf.matmul(self.obs,self.w)

            self.target = tf.placeholder(
                tf.float32,shape=[None,action_shape],name="%s_target"%scope
            )
            taction = tf.transpose(tf.transpose(self.target) / tf.reduce_sum(self.target, axis=1))
            tloss = taction * (self.output - self.target)
            self.loss = tf.reduce_sum(tloss * tloss)
            self.optimizer = tf.train.GradientDescentOptimizer(0.001)
            self.ttrain = self.optimizer.minimize(self.loss)
        self.sess = sess
        self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def train(self, state_set, target_set):
        self.sess.run(self.ttrain,{self.obs:state_set,self.target:target_set})
