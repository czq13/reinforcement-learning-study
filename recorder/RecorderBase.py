
import numpy as np
from gui.rl_gui import RLGui
from parameters import pms
class RecorderBase():
    def __init__(self):
        self.intend_data = {pms.agent_name:[['train_avg_reward'],['train_total_reward'],['test_avg_reward'],['test_total_reward']]}
        self.gui = None
        self.data_map = {}
        #self.plot_map()
        pass

    def plot_map(self):
        self.gui = RLGui(self.intend_data)
        for k in self.intend_data.keys():
            for i in self.intend_data[k]:
                for j in i:
                    self.data_map[j] = []
    def show(self):
        for k in self.intend_data.keys():
            line_map = {}
            for i in self.intend_data[k]:
                line_map[i[0]] = {}
                for j in i:
                    if (type(self.data_map[j]) == list):
                        line_map[i[0]][j] = [range(len(self.data_map[j])),self.data_map[j]]
                    else:
                        line_map[i[0]][j] = [range(len(self.data_map[j])), self.data_map[j].tolist()]
            self.gui.update_line(k,line_map)

    def add_data(self,vlist,vdata):
        for i in range(len(vlist)):
            self.data_map[vlist[i]].append(vdata[i])
        self.show()

    def set_data(self,vlist,vdata):
        for i in range(len(vlist)):
            self.data_map[vlist[i]] = vdata[i]
        self.show()

