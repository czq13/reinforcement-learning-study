import tensorflow as tf
from NetworkBase import NetworkBase
from parameters import pms

class NetworkNet(NetworkBase):
    def __init__(self,scope,sess,obs_shape = pms.system_state_num,action_shape = pms.system_action_num):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = tf.placeholder(
                tf.float32,shape=[None,obs_shape],name="%s_obs"%scope
            )
            #self.w = tf.get_variable('Matrix',[obs_shape,action_shape],tf.float32,
            #                         tf.random_normal_initializer(stddev=0.02))
            layer1_w = tf.get_variable('layer1_w',[obs_shape,64],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer1_b = tf.get_variable('layer1_b',[64],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer2_w = tf.get_variable('layer2_w',[64,64],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer2_b = tf.get_variable('layer2_b',[64],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer3_w = tf.get_variable('layer3_w',[64,action_shape],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer3_b = tf.get_variable('layer3_b',[action_shape],tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
            layer1_output = tf.tanh(tf.matmul(self.obs,layer1_w)+layer1_b)
            layer2_output = tf.tanh(tf.matmul(layer1_output,layer2_w)+layer2_b)
            layer3_output = tf.matmul(layer2_output,layer3_w) + layer3_b
            self.output = layer3_output

            self.target = tf.placeholder(
                tf.float32,shape=[None,action_shape],name="%s_target"%scope
            )
            taction = tf.transpose(tf.transpose(self.target)/tf.reduce_sum(self.target,axis=1))
            tloss = taction * (self.output - self.target)
            self.loss = tf.reduce_sum(tloss * tloss)
            self.optimizer = tf.train.GradientDescentOptimizer(pms.gradient_step)
            self.ttrain = self.optimizer.minimize(self.loss)
        self.sess = sess
        self.var_list = [v for v in tf.trainable_variables() if v.name.startswith(scope)]

    def train(self, state_set, target_set):
        self.sess.run(self.ttrain,{self.obs:state_set,self.target:target_set})
