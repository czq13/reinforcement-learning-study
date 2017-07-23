import time
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from line_plotter import LinePlotter
#inner loop means number loops that use samples to train network
#outer loop means get samples then go inner loop
class RLGui(object):
    def __init__(self,record_map):
        self.fig_map = {}
        self.fig_list = {}
        self.time_str = time.strftime('%Y_%m_%d',time.localtime(time.time()))
        for k in record_map.keys():
           self.add_fig(k,record_map[k])

    def add_fig(self,map_name,name_list):
        plt.ion()
        _fig = plt.figure(figsize=(14, 14))
        _fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        _gs1 = gridspec.GridSpec(19, 19)
        self.fig_map[map_name] = {}
        for i in range(len(name_list)):
            if i < 2:
                c = LinePlotter(_fig, _gs1[1:9, 1+9*i:9+9*i], name_list[i][0], name_list[i])
            else:
                c = LinePlotter(_fig, _gs1[10:18, 1 + 9 * (i-2):9 + 9 * (i-2)], name_list[i][0], name_list[i])
            self.fig_map[map_name][name_list[i][0]] = c
        _fig.canvas.draw()
        self.fig_list[map_name] = _fig

    def update_line(self,map_name,lines):
        for i in lines.keys():
            #if len(lines[i])
            self.fig_map[map_name][i].update_lines(lines[i])

    def save_fig(self):
        for i in self.fig_list.keys():
            name_str = 'fig_result/' + i + self.time_str
            self.fig_list[i].savefig(name_str)
