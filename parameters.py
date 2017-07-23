import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('environment_name','CartPole-v0','environment name')
flags.DEFINE_integer('system_state_num',4,'state num')
flags.DEFINE_integer('system_action_num',2,'action num')

flags.DEFINE_string('network_name','NetworkLinear','network name')
#flags.DEFINE_string('network_name','NetworkNet','network name')
flags.DEFINE_string('train_name','TrainBase','train way to set replay method')
flags.DEFINE_string('agent_name','AgentBase','the agent name')
flags.DEFINE_string('recorder_name','RecorderQFunc','the agent name')

flags.DEFINE_integer('history_len',20000,'replay history lenth')
flags.DEFINE_integer('batch_size',250,'batch size')
flags.DEFINE_float('gamma',0.95,'the learning rate in q learning')
flags.DEFINE_integer('return_type',0,'scale state num or not')

flags.DEFINE_integer('maxlen',2000,'the max len per epoch')
flags.DEFINE_float('epsilon',0.5,'the epsilon-greedy to choose action')

flags.DEFINE_bool('discrete',True,'decide the plane is discrete or not')

flags.DEFINE_integer('epoch_num',2000,'how much epoch in agent')
flags.DEFINE_integer('sync_cycle',2,'sync net cycle')

flags.DEFINE_bool('update_limit',True,'decide if the env update obs limit')
flags.DEFINE_integer('test_frequence',2,'test frequence')
flags.DEFINE_float('gradient_step',0.001,'gradient step')
flags.DEFINE_string('checkpoint_dir','checkpoint/','where is checkpoint')
flags.DEFINE_integer('save_model_fre',5,'save model frequence')
pms = flags.FLAGS