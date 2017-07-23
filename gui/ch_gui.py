import time
import threading
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


from experiments.gui.config import config
from experiments.gui.action_panel import Action, ActionPanel
from experiments.gui.textbox import Textbox
from experiments.gui.line_plotter import LinePlotter
#inner loop means number loops that use samples to train network
#outer loop means get samples then go inner loop
class CHGUI(object):
    def __init__(self):
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
        #13: test
        #14: zoom in
        #15: zoom out
        #16: opt2backup
        #17: backup2opt
        self.Instruction = 0
        self.outer_steps = 0
        self.inner_steps = 0
        # this outer means fixing action network
        # inner means fixing q network
        # total inner 49 * 10
        # outer loop every one step test once
        actions_arr = [
            Action('outer_1', 'outer_1', self.set_outer_one, axis_pos=0),
            Action('outer_5', 'outer_5', self.set_outer_five, axis_pos=1),
            Action('opt2backup', 'opt2backup', self.opt2backup, axis_pos=2),
            Action('resample', 'resample', self.resample, axis_pos=3),
            Action('inner_5', 'inner_5', self.set_inner_five, axis_pos=4),
            Action('backup2opt', 'backup2opt', self.backup2opt, axis_pos=5),
            Action('zoomin', 'zoomin', self.set_zoom_in, axis_pos=6),
            Action('zoomout', 'zommout', self.set_zoom_out, axis_pos=7),
            Action('outer_continue', 'outer_continue', self.set_outer_continue, axis_pos=8),
            Action('inner_continue', 'inner_continue', self.set_inner_continue, axis_pos=9),#means go on finish 30 itr,if > 30, try outer
            Action('reset_0', 'reset_0', self.reset_parameter, axis_pos=10),
            Action('test_action', 'test_action', self.test_action, axis_pos=11),
        ]
        plt.ion()
        self._fig = plt.figure(figsize=config['figsize'])
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                  wspace=0, hspace=0)
        self._gs = gridspec.GridSpec(33, 18)
        self._gs_action_panel = self._gs[0:4, 0:12]
        self._gs_action_output = self._gs[0:4, 12:18]
        self._gs_q_value = self._gs[5:18,1:9]
        self._gs_q_loss = self._gs[5:18,10:18]
        self._gs_test_loss = self._gs[19:32,1:9]
        self._gs_q1 = self._gs[19:32,10:18]

        self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 2, 6, actions_arr)
        self._action_output = Textbox(self._fig, self._gs_action_output, border_on=True)
        self._q_value = LinePlotter(self._fig,self._gs_q_value,'target_q',['q_value','target_q'])
        self._q_loss = LinePlotter(self._fig,self._gs_q_loss,'test_loss',['test_loss'])
        self._test_loss = LinePlotter(self._fig,self._gs_test_loss,'test_reward',['test_reward'])
        self._q1 = LinePlotter(self._fig,self._gs_q1,'test_q1',['q_value', 'target_q'])

        self.colors = ['cyan','orange']
        self.color_state = 0

        self._fig.canvas.draw()

        self._fig2 = plt.figure(figsize=config['figsize'])
        self._fig2.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                   wspace=0, hspace=0)
        self._gs2 = gridspec.GridSpec(19, 19)
        self._q2 = LinePlotter(self._fig2, self._gs2[1:9, 1:9], 'test_q2', ['q_value', 'target_q'])
        self._q3 = LinePlotter(self._fig2, self._gs2[1:9, 10:18], 'test_q3', ['q_value', 'target_q'])
        self._q4 = LinePlotter(self._fig2, self._gs2[10:18, 1:9], 'test_q4', ['q_value', 'target_q'])
        self._q5 = LinePlotter(self._fig2, self._gs2[10:18, 10:18], 'test_q5', ['q_value', 'target_q'])
        self._fig2.canvas.draw()

        self._fig3 = plt.figure(figsize=config['figsize'])
        self._fig3.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                   wspace=0, hspace=0)
        self._gs3 = gridspec.GridSpec(19, 19)
        self._a1 = LinePlotter(self._fig3, self._gs3[1:9, 1:9], 'test_a1', ['thetad', 'theta'])
        self._a2 = LinePlotter(self._fig3, self._gs3[1:9, 10:18], 'test_a2', ['thetad', 'theta'])
        self._a3 = LinePlotter(self._fig3, self._gs3[10:18, 1:9], 'test_a3', ['thetad', 'theta'])
        self._a4 = LinePlotter(self._fig3, self._gs3[10:18, 10:18], 'test_a4', ['thetad', 'theta'])
        self._fig3.canvas.draw()


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
    def test_action(self,event=None):
        self.Instruction = 13
        self.new_flag = True
    def set_zoom_in(self,event=None):
        self.Instruction = 14
        self.new_flag = True
    def set_zoom_out(self,event=None):
        self.Instruction = 15
        self.new_flag = True
    def opt2backup(self,event = None):
        self.Instruction = 16
        self.new_flag = True
    def backup2opt(self,event = None):
        self.Instruction = 17
        self.new_flag = True
    def resample(self,event = None):
        self.Instruction = 18
        self.new_flag = True
    def process_Instruction(self,dict_info,algorithm):
        #this function should be called by gpsmain
        #parameter reset
        if self.Instruction == 7:
            algorithm.reset_q_net_var()
        elif self.Instruction == 14:
            algorithm.step /= 2.0
        elif self.Instruction == 15:
            algorithm.step *= 2.0
        elif self.Instruction == 16:
            algorithm.policy_opt.opt_to_backup_network()
        elif self.Instruction == 17:
            algorithm.policy_opt.backup_to_opt_network()
        #change test_coler
        self.color_state = (self.color_state + 1) % 2
        self.set_action_bgcolor(self.colors[self.color_state], alpha=1.0)
        #change text
        self.show_Info(dict_info)
        self.new_flag = False

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)

    def show_Info(self,dict_info):
        inner = 'inner:\nstep:%2d,done:%2d,total:%2d' % (
        dict_info['istep'], dict_info['ialready_done'], dict_info['itotal'])
        outer = 'outer:\nstep:%2d,done:%2d,total:%2d' % (
        dict_info['ostep'], dict_info['oalready_done'], dict_info['ototal'])
        self._action_output.set_text(inner)
        self._action_output.append_text(outer)
    def update_q_lines(self,new_lines):
        self._q_value.update_lines(new_lines[0])
        self._q_loss.update_lines(new_lines[7])
        self._q1.update_lines(new_lines[2])
        self._q2.update_lines(new_lines[3])
        self._q3.update_lines(new_lines[4])
        self._q4.update_lines(new_lines[5])
        self._q5.update_lines(new_lines[6])
    def update_test_action(self,new_lines):
        self._a1.update_lines(new_lines[0])
        self._a2.update_lines(new_lines[1])
        self._a3.update_lines(new_lines[2])
        self._a4.update_lines(new_lines[3])
       # self._test_loss.update_lines(new_lines[5])