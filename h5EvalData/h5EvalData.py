# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:04:32 2021

@author: jana
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic as bs
import h5py

#---------Helper functions-----------------------------------------------------
def create_bins(bins):
    bins_diff = np.diff(bins)
    leftpad = np.array([bins[0]-bins_diff[0]])
    rightpad = np.array([bins[-1]+bins_diff[-1]])
    bins_pad = np.concatenate((leftpad, bins, rightpad))
    bins_new = bins_pad[:-1] + np.diff(bins_pad)/2
    return bins_new

def bin_data(x, y, bins=None):
    x_mean = bs(x, x, statistic='mean', bins=bins)[0]
    y_mean = bs(x, y, statistic='mean', bins=bins)[0]
    
    return x_mean, y_mean
    
def fig_kwargs(kwargs):
    figsize = kwargs.get('figsize', plt.rcParams['figure.figsize'])
    title = kwargs.get('title', '')
    ls = kwargs.get('ls', '')
    return figsize, title, ls
#------------------------------------------------------------------------------
class h5Evaluation:
    '''
    '''
    def __init__(self, filename, filepath, ext, motor='Pt_No', chnl='epoch'):
        self.h5 = h5py.File(filepath+filename+ext,'r')
        self.motor = motor
        self.chnl = chnl
    
    def data_matrix(self, sl): # not in use, works when all scans are of same length
        self.sl = sl
        sn0 = self.sl[0]
        key_list = list(self.h5['entry%d'%sn0]['measurement'].keys())
        keysToRemove = ['pre_scan_snapshot', 'ref', 'refM', 'rel', 'relM', 
                        'spec', 'specM']
        key_list = [keys for keys in key_list if keys not in keysToRemove]
        self.keys = key_list
        m = len(key_list)
        n = len(self.sl)
        Pt_No = self.h5['entry%d'%sn0]['measurement']['Pt_No'].shape[0]
        data_mat = np.zeros((m,n,Pt_No))
        for i, key in enumerate(self.keys):
            for j, sn in enumerate(self.sl):
                data_mat[i, j, :] = self.h5['entry%d'%sn]['measurement'][key][()]
        self.data_mat = data_mat
    
    def mean(self, sl): # not in use, works when all scans are of same length
        self.data_matrix(sl)
        idx_motor = self.keys.index(self.motor)
        tf_chnl = []
        for chnl in self.keys:
            if chnl in self.chnl:
                tf_chnl.append(True)
            else:
                tf_chnl.append(False)
        self.x = self.data_mat[idx_motor][0]
        self.y_mean = self.data_mat[tf_chnl].mean(axis=1)
    
    def data_dict(self, sl):
        '''
        Create a data dictionary. The channel names are set as the keys. Measured
        data for all scans corresponds to each channel are concatenated and assign
        to the corresponding key.
        '''
        self.sl = sl
        sn0 = self.sl[0]
        key_list = list(self.h5['entry%d'%sn0]['measurement'].keys())
        keysToRemove = ['pre_scan_snapshot', 'ref', 'refM', 'rel', 'relM', 
                        'spec', 'specM']
        key_list = [keys for keys in key_list if keys not in keysToRemove]
        self.keys = key_list
        data_dic = {}
        for i, key in enumerate(self.keys):
            data_conc = np.array([])
            for j, sn in enumerate(self.sl):
                data_sn = self.h5['entry%d'%sn]['measurement'][key][()]
                data_conc = np.concatenate((data_conc, data_sn))
            data_dic[key] = data_conc
        x = np.array([])
        for i, sn in enumerate(self.sl):
            tem_x = self.h5['entry%d'%sn]['measurement'][self.motor][()]
            if len(tem_x)>len(x):
                x = tem_x
        data_dic['x'] = x
        self.data_dic = data_dic
    
    def data_stat(self, sl, bins=None):
        '''
        Calculates mean, std, sum (more can be implemented) for each desired channels
        for a scan list (sl) after binning to bins. When bins is None, it takes the 
        longest x axis (the longest range of the motor) to create an appropriate bins.

        Parameters
        ----------
        sl : list of integers
            list of the scan numbers.
            
        bins : int or sequence of scalars, optional (default None)
            If bins is an int, it defines the number of equal-width bins in 
        the given range. If bins is a sequence, it defines the bin edges, including 
        the rightmost edge, allowing for non-uniform bin widths. Values in x that 
        are smaller than lowest bin edge are assigned to bin number 0, values 
        beyond the highest bin are assigned to bins[-1]. If the bin edges are 
        specified, the number of bins will be, (nx = len(bins)-1).

        '''
        self.data_dict(sl)
        if bins is None:
            bins = create_bins(self.data_dic['x'])
        x = self.data_dic[self.motor]
        y = np.zeros((len(self.chnl), len(x)))
        for i, c in enumerate(self.chnl):
                       y[i, :] = self.data_dic[c]
        
        x_mean = bs(x, x, statistic='mean', bins=bins)[0]
        y_mean = bs(x, y, statistic='mean', bins=bins)[0]
        x_std = bs(x, x, statistic='std', bins=bins)[0]
        y_std = bs(x, y, statistic='std', bins=bins)[0]
        x_sum = bs(x, x, statistic='sum', bins=bins)[0]
        y_sum = bs(x, y, statistic='sum', bins=bins)[0]
        self.xmean = x_mean
        self.ymean = y_mean
        self.xstd = x_std
        self.ystd = y_std
        self.xsum = x_sum
        self.ysum = y_sum
        
    def plot_mean(self, sl, kwargs):
        #self.mean(sl) # works when all scans are of same length
        self.data_stat(sl)
        #colors = rcParams["axes.prop_cycle"]()
        figsize, title, ls = fig_kwargs(kwargs)
        plt.figure(figsize=figsize)
        lineObjects = plt.plot(self.xmean, self.ymean.T, 'o', ls=ls)
        #lineObjects = plt.plot(self.x, self.y_mean.T, 'o', ls='--') # works when all scans are of same length
        plt.legend(lineObjects, self.chnl, frameon=False)
        plt.xlabel(self.motor)
        plt.ylabel('measurement')
        # plt.axhline(color='k', ls='--', lw=1.0)
        # plt.axvline(color='k', ls='--', lw=1.0)
        plt.title(title)
        plt.show()
    
    def plot_seq(self, seq, kwargs):
        figsize, title, ls = fig_kwargs(kwargs)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        legends = []
        for i in range(len(seq)):
            sl = seq[i][0]
            self.data_stat(sl)
            ax.plot(self.xmean, self.ymean.T, 'o', ls=ls)
            for j in range(len(self.chnl)):
                legends.append(self.chnl[j] + '_' + str(seq[i][1]))
        ax.legend(legends, frameon=False)
        ax.set_xlabel(self.motor)
        ax.set_ylabel('measurement')
        plt.show()

    def plot_seq_norm(self, seq, kwargs):
        figsize, title, ls = fig_kwargs(kwargs)
        pass
        
    def close(self):
        self.h5.close()
        
        
        
        