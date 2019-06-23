# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:49:46 2019

@authors:
    Alberto Fossa'
    Giuliana Elena Miceli
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

class matlab_interface:
    
    def __init__(self, data_container, mat_file=True,
                 keys=('Isp', 'twr', 'prop_frac', 'tof', 'alpha0', 'failures', 'fail_summary', 'Isp_fail', 'twr_fail')):
        
        """
        initialize a new matlab_interface instance loading the Matlab results
        saved in the specified .mat file or dictionary
        
        input:
            data_container:     .mat file or dictionary from which the data are loaded
            keys:               tuple with the dictionary keys used in data_container
            mat_file:           load data from .mat file (True) or from a dictionary (False)
        """
        
        if mat_file:
            self.data = loadmat(data_container, squeeze_me=True, mat_dtype=True)
        else:
            self.data = data_container
        self.keys = keys
                        
    def set_array_lim(self, k, lim):
        
        """
        extracts from the array specified by the key k the values that fall between
        the limits specified in lim as (min, max)
        
        input:
            k:      key corresponding to the array to modify
            lim:    limiting values (min, max)
            
        output:
            arr_lim:    limited copy of the array k
            idx_lim:    indexes in the array k corresponding to the limits in lim
        """
        
        arr = self.data[k]
        idx_all = np.nonzero((arr>=lim[0]) & (arr<=lim[1]))[0]
        idx_lim = (idx_all[0], idx_all[-1])
        arr_lim = arr[idx_lim[0]:idx_lim[1]+1]
        
        return arr_lim, idx_lim
    
    def set_data_lim(self, Isp_lim, twr_lim):
        
        """
        updates the data dictionary taking only the values corresponding to the Isp and twr
        values limited by Isp_lim and twr_lim, where:
            
            Isp_lim: (Isp_min, Isp_max)
            twr_lim: (twr_min, twr_max)
        """
        
        self.data[self.keys[0]], Isp_idx = self.set_array_lim(self.keys[0], Isp_lim)
        self.data[self.keys[1]], twr_idx = self.set_array_lim(self.keys[1], twr_lim)
        
        for i in range(2,5):
            self.data[self.keys[i]] = self.data[self.keys[i]][twr_idx[0]:twr_idx[1]+1, Isp_idx[0]:Isp_idx[1]+1]
            
    def get_data(self):
        
        """
        returns a dictionary with all the data corresponding to the defined limits
        """
        
        return self.data
    
    def get_opt_data(self):
        
        i = self.data['idx_opt'][0]
        j = self.data['idx_opt'][1]
                
        d = {}
        
        d['H'] = self.data['H']
        d['rf'] = self.data['rf']
        
        d['rf'] = 1737.4e3*1.05
        
        d['Isp'] = self.data['Isp'][j]
        d['twr'] = self.data['twr'][i]
        d['tof'] = self.data['tof'][i,j]
        d['m0'] = self.data['m0']
        d['m_final'] = self.data['m_final'][i,j]
        d['m_prop'] = self.data['m_prop'][i,j]
        d['prop_frac'] = self.data['prop_frac'][i,j]
        d['alpha0'] = self.data['alpha0'][i,j]
        d['hist'] = self.data['hist']
        
        return d
        
    def get_Isp_twr(self):
        
        """
        returns the Isp and twr arrays
        """
        
        return self.data[self.keys[0]], self.data[self.keys[1]]
    
    def get_lim(self):
        
        """
        returns two tuples corresponding to the current Isp and twr limits defined as:
            
            Isp_lim: (Isp_min, Isp_max)
            twr_lim: (twr_min, twr_max)
        """
        Isp = self.data[self.keys[0]]
        twr = self.data[self.keys[1]]
        Isp_lim = (Isp[0], Isp[-1])
        twr_lim = (twr[0], twr[-1])
        
        return Isp_lim, twr_lim
    
    def get_errors(self, other, display=False):
        
        """
        returns the maximum absolute and relative errors and the corresponding indexes
        between the propellant fraction, time of flight and initial thrust angle matrices
        of the self and the other Matlab interface instances
        
        input:
            other: the Matlab instance between which the errors are computed
            display: print (True) or not (False) the obtained errors
            
        output:
            err: dictionary with three tuples of (max_abs_err, max_rel_err, max_err_idx)
        """
        err={}
        
        for i in range(2,5):
            
            abs_err = np.abs(self.data[self.keys[i]]-other.data[other.keys[i]]) #absolute errors matrix
            abs_max_err = np.nanmax(abs_err) #maximum absolute error
            idx = np.unravel_index(np.nanargmax(abs_err), np.shape(abs_err)) #maximum absolute error indexes
            rel_max_err = abs_max_err/self.data[self.keys[i]][idx[0],idx[1]] #maximum relative error
            
            err[self.keys[i]] = (abs_max_err, rel_max_err, idx)
            
        if display:
            
            print("\nMaximum errors between the solutions:")
            print("\nPropellant fraction:")
            print("absolute: " + str(err[self.keys[2]][0]) + "\nrelative: " + str(err[self.keys[2]][1]))
            print("\nTime of flight:")
            print("absolute: " + str(err[self.keys[3]][0]) + " s\nrelative: " + str(err[self.keys[3]][1]))
            print("\nInitial thrust angle:")
            print("absolute: " + str(err[self.keys[2]][0]*180/np.pi) + " deg\nrelative: " + str(err[self.keys[4]][1]))
        
        return err
    
    def get_failures(self, display=False):
        
        """
        prints a summary of the encountered optimizer failures and returns the corresponding Isp and twr arrays
        """
        
        if display:
            print("\nNumber of runs: " + str(self.data[self.keys[5]][0]))
            print("\nNumber of failures: " + str(self.data[self.keys[5]][1]))
            print("\nFailure rate: " + str(self.data[self.keys[5]][2]))
        
        if self.data[self.keys[5]][1]>0: #at least one failure
            return self.data[self.keys[6]], self.data[self.keys[7]]
        
        else:
            return 0, 0
        
    def data_contour(self, k, title, scale=1.0):
        
        """
        plots a contour of the data specified by the key k as a function of twr and Isp
        scaled by the quantity scale
        """
                    
        [X,Y]=np.meshgrid(self.data[self.keys[1]], self.data[self.keys[0]], indexing='ij')
        
        fig, ax = plt.subplots()
        cs = ax.contour(X,Y,self.data[k]*scale, 25)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title(title)
        
    def generic_contour(self, z, title, scale=1.0):
        
        """
        plots a contour of the data specified by the key k as a function of twr and Isp
        scaled by the quantity scale
        """
                    
        [X,Y]=np.meshgrid(self.data[self.keys[1]], self.data[self.keys[0]], indexing='ij')
        
        fig, ax = plt.subplots()
        cs = ax.contour(X,Y,z*scale, 25)
        ax.clabel(cs)
        ax.set_xlabel('Thrust/initial weight ratio')
        ax.set_ylabel('Isp (s)')
        ax.set_title(title)