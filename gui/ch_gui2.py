import time
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.gui.config import config
from experiments.gui.action_panel import Action, ActionPanel
from experiments.gui.textbox import Textbox
from experiments.gui.line_plotter import LinePlotter
#inner loop means number loops that use samples to train network
#outer loop means get samples then go inner loop
class CHGUI(object):
    def __init__(self,):
        #self._hyperparams = hyperparams
        self.new_flag = False

        #0 : no instruction
        #1 : outer loop move forward one step
        #2 : outer loop move forward five step
        #3 : outer loop move forward ten stap
        #4 : inner loop move forward one step
        #5 : inner loop move forward five step
        #6 : inner loop move forward ten step
        #TODO : set total outer loop step and inner loop step
        #7 : reset parameter
        #8 : reset all parameter
        #9 : outer continue execute
        #10: stop
        #11: inner continue execute
        #12: save fig
        #13: test all data
        #14: give the w_q to w_q_pred
        self.Instruction = 0
        self.outer_steps = 0
        self.inner_steps = 0

        actions_arr = [
            Action('inner_1', 'inner_1', self.set_inner_one, axis_pos=0),
            Action('inner_5', 'inner_5', self.set_inner_five, axis_pos=1),
            Action('inner_10', 'inner_10', self.set_inner_ten, axis_pos=2),
            Action('inner_continue', 'inner_continue', self.set_inner_continue, axis_pos=3),#means go on finish 30 itr,if > 30, try outer
            Action('test', 'test', self.test_all_data, axis_pos = 4),
            Action('change', 'change', self.change_q_value, axis_pos = 5)
        ]
        plt.ion()
        self._fig = plt.figure(figsize=config['figsize'])
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                  wspace=0, hspace=0)
        self._gs = gridspec.GridSpec(33, 18)
        self._gs_action_panel = self._gs[0:4, 0:12]
        self._gs_action_output = self._gs[0:4, 12:18]
        self._gs_q_error = self._gs[5:18,1:9]
        self._gs_loss = self._gs[5:18,10:18]
        self._gs_test = self._gs[19:32,1:9]
        self._gs_target_q_5 = self._gs[19:32,10:18]

        self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 1, 6, actions_arr)
        self._action_output = Textbox(self._fig, self._gs_action_output, border_on=True)
        self._q_repaly = LinePlotter(self._fig,self._gs_q_error,'replay',['q_value','target_q'])
        self._loss = LinePlotter(self._fig,self._gs_loss,'loss',['replay_loss'])
        self._test = LinePlotter(self._fig,self._gs_test,'test_loss',['test_loss'])
        self._target_q_5 = LinePlotter(self._fig,self._gs_target_q_5,'5',['q_value','target_q'])

        self.colors = ['cyan','orange']
        self.color_state = 0

        self._fig.canvas.draw()

        self._fig2 = plt.figure(figsize=config['figsize'])
        self._fig2.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                  wspace=0, hspace=0)
        self._gs2 = gridspec.GridSpec(19,19)
        self._q_targetq_1 = LinePlotter(self._fig2,self._gs2[1:9,1:9],'1',['q_value','target_q'])
        self._q_targetq_2 = LinePlotter(self._fig2,self._gs2[1:9,10:18],'2',['q_value','target_q'])
        self._q_targetq_3 = LinePlotter(self._fig2,self._gs2[10:18,1:9],'3',['q_value','target_q'])
        self._q_targetq_4 = LinePlotter(self._fig2,self._gs2[10:18,10:18],'4',['q_value','target_q'])
        self._fig2.canvas.draw()


    def set_outer_one(self, event=None):
        self.Instruction = 1
        self.outer_steps = 1
        self.new_flag = True

    def set_outer_five(self, event=None):
        self.Instruction = 2
        self.outer_steps = 5
        self.inner_steps = -1
        self.new_flag = True

    def set_outer_ten(self, event=None):
        self.Instruction = 3
        self.outer_steps = 10
        self.inner_steps = -1
        self.new_flag = True

    def set_inner_one(self, event=None):
        self.Instruction = 4
        self.inner_steps = 1
        self.new_flag = True

    def set_inner_five(self, event=None):
        self.Instruction = 5
        self.inner_steps = 5
        self.new_flag = True

    def set_inner_ten(self, event=None):
        self.Instruction = 6
        self.inner_steps = 10
        self.new_flag = True

    def reset_parameter(self, event=None):
        self.Instruction = 7
        self.new_flag = True

    def reset_all_parameter(self, event=None):
        self.Instruction = 8
        self.new_flag = True

    def set_outer_continue(self, event=None):
        self.Instruction = 9
        self.outer_steps = -1
        self.inner_steps = -1
        self.new_flag = True

    def set_stop(self, event=None):
        self.Instruction = 10
        self.new_flag = True

    def set_inner_continue(self, event=None):
        self.Instruction = 11
        self.inner_steps = -1
        self.new_flag = True
    def save_fig(self,event=None):
        self.Instruction = 12
        tmp = '/home/czq13/chWorkspace/gps_6/experiments/pitch_control/save'
        plt.savefig(tmp, dpi=75)
        self.new_flag = True
    def test_all_data(self,event=None):
        self.Instruction = 13
        self.new_flag = True
    def change_q_value(self,event=None):
        self.Instruction = 14
        self.new_flag = True
    def process_Instruction(self,dict_info,algorithm):
        #this function should be called by gpsmain
        #parameter reset
        if self.Instruction == 7:
            algorithm.reset_q_net_var()
        elif self.Instruction == 14:
            algorithm.policy_opt.update_target_q_network()
        #change test_coler
        self.color_state = (self.color_state + 1) % 2
        self.set_action_bgcolor(self.colors[self.color_state], alpha=1.0)
        #change text
        self.show_Info(dict_info)
        self.new_flag = False

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)

    def show_Info(self,dict_info):
        inner = 'step:%2d,done:%2d,total:%2d' % (
        dict_info['istep'], dict_info['ialready_done'], dict_info['itotal'])
        self._action_output.set_text(inner)
    def update_lines(self,new_lines):
        self._q_repaly.update_lines(new_lines[0])
        self._loss.update_lines(new_lines[1])

    def update_five_lines(self,lines):
        self._q_targetq_1.update_lines(lines[0])
        self._q_targetq_2.update_lines(lines[1])
        self._q_targetq_3.update_lines(lines[2])
        self._q_targetq_4.update_lines(lines[3])
        self._target_q_5.update_lines(lines[4])
        self._test.update_lines(lines[5])