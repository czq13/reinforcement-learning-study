import time
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from line_plotter import LinePlotter
#inner loop means number loops that use samples to train network
#outer loop means get samples then go inner loop
class CHGUI(object):
    def __init__(self,):

        plt.ion()
        self._fig1 = plt.figure(figsize=(14,14))
        self._fig1.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,wspace=0, hspace=0)
        self._gs1 = gridspec.GridSpec(19,19)
        self.c1 = LinePlotter(self._fig1, self._gs1[1:9, 1:9], '1', ['action'])
        self.c2 = LinePlotter(self._fig1, self._gs1[1:9, 10:18], '2', ['wind_force'])
        self.c3 = LinePlotter(self._fig1, self._gs1[10:18, 1:9], '3', ['train_reward'])
        self.c4 = LinePlotter(self._fig1, self._gs1[10:18, 10:18], '4', ['test_reward'])
        self._fig1.canvas.draw()
        '''
        self._fig2 = plt.figure(figsize=(14,14))
        self._fig2.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,wspace=0, hspace=0)
        self._gs2 = gridspec.GridSpec(19,19)
        self.s1 = LinePlotter(self._fig2,self._gs2[1:9,1:9],'1',['x_earth_err'])
        self.s2 = LinePlotter(self._fig2,self._gs2[1:9,10:18],'2',['y_earth_err'])
        self.s3 = LinePlotter(self._fig2,self._gs2[10:18,1:9],'3',['z_earth_err'])
        self.s4 = LinePlotter(self._fig2,self._gs2[10:18,10:18],'4',['wind_body'])
        self._fig2.canvas.draw()

        self._fig3 = plt.figure(figsize=(14, 14))
        self._fig3.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        self._gs3 = gridspec.GridSpec(19, 19)
        self.ss1 = LinePlotter(self._fig3, self._gs3[1:9, 1:9], '1', ['theta'])
        self.ss2 = LinePlotter(self._fig3, self._gs3[1:9, 10:18], '2', ['psi'])
        self.ss3 = LinePlotter(self._fig3, self._gs3[10:18, 1:9], '3', ['u_body'])
        self.ss4 = LinePlotter(self._fig3, self._gs3[10:18, 10:18], '4', ['v_body'])
        self._fig2.canvas.draw()
        '''
    def updateLine0(self,lines):
        self.c1.update_lines(lines)
    def updateLine1(self,lines):
        self.c2.update_lines(lines)
    def updateLine2(self,lines):
        self.c3.update_lines(lines)
    def updateLine3(self,lines):
        self.c4.update_lines(lines)
'''
    def updateLine2(self,lines):
        self.s1.update_lines(lines[0])
        self.s2.update_lines(lines[1])
        self.s3.update_lines(lines[2])
        self.s4.update_lines(lines[3])

    def updateLine3(self,lines):
        self.ss1.update_lines(lines[0])
        self.ss2.update_lines(lines[1])
        self.ss3.update_lines(lines[2])
        self.ss4.update_lines(lines[3])
'''