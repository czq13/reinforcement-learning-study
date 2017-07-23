from gui.rl_gui import RLGui
from parameters import pms
from RecorderBase import RecorderBase

class RecorderQFunc(RecorderBase):
    def __init__(self):
        #super(RecorderQFunc,self).__init__()
        RecorderBase.__init__(self)
        self.intend_data['q_watcher'] = [['train_q0','train_q1'],['test_q0','test_q1'],['train_action'],['test_action']]
        self.plot_map()