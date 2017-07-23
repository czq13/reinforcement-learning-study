
import matplotlib.pyplot as plt
from agent.AgentBase import AgentBase
from agent.AgentSVM import AgentSVM
from parameters import pms
plt.switch_backend('TKagg')

agent = eval(pms.agent_name)()
agent.train_net()
agent.recorder.gui.save_fig()
'''
import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32,shape=[None,2])
w = tf.get_variable('w',[2,2],tf.float32)
b = tf.matmul(a,w)
c = tf.reduce_sum(b,axis=1)
d = tf.transpose(tf.transpose(b) / c)
e = tf.reduce_sum(b)
f = d * b
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print sess.run(b,{a:np.array([[1,2],[3,4],[5,6]])})
print sess.run(c,{a:np.array([[1,2],[3,4],[5,6]])})
print sess.run(d,{a:np.array([[1,2],[3,4],[5,6]])})
print sess.run(f,{a:np.array([[1,2],[3,4],[5,6]])})

from network.svmlib.svmutil import *
import numpy as np
x = np.random.rand(100,2)
y = x*x
prom = svm_problem(y.tolist(),x.tolist())
param = svm_parameter('-s 4 -t 2 -c 4')
m = svm_train(prom,param)
t1,t2,t3 = svm_predict(y.tolist(),x.tolist(),m)
print t1
print t2
print t3
print y.tolist()
print sess.run(e,{a:np.array([[1,2],[3,4],[5,6]])})
'''