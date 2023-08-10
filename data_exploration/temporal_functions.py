## Author: Jaleesa Houle

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

import astropy
from astropy import stats
import math

import scipy
from scipy import stats

from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, atleast_1d, sqrt, array,
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)


from scipy import optimize
from scipy import special



def maxchange_std (df, startval, lagarray, method='forward'):
    
    d_list=np.zeros((len(lagarray)))
    j=0
    
    def checkforwardbounds(df,startval,lag):
        if (startval+i)>=len(df):
            raise ValueError('Lag value extends beyond length of timeseries')            
    
    def checkbackwardbounds(df,startval,lag):
        if (startval-i)<0:
            raise ValueError('Lag value extends beyond length of timeseries')  

    
    for i in lagarray:
        
        if method=='forward':
            checkforwardbounds(df,startval,i)
            b=df.iloc[startval:startval+i]
            std= mycircstd(b, high=360, low=0) 
            d_list[j]=std
                
        j+=1
    return d_list


def _circfuncs_common(samples, high, low, nan_policy='propagate'):
    # Ensure samples are array-like and size is not zero
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = sin((samples - low)*2.*pi / (high - low))
    cos_samp = cos((samples - low)*2.*pi / (high - low))
    #sin_samp=sin(samples)
    #cos_samp=cos(samples)
    mask = None
    return samples, sin_samp, cos_samp, mask


def mycircstd(samples, high=360, low=0, axis=None, nan_policy='propagate'):

    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)  # [1] (2.2.3)
        cos_mean = cos_samp.mean(axis=axis)  # [1] (2.2.3)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))  # [1] (2.2.4)

    #res = sqrt(-2*log(R))
    #if not normalize:
    #    res *= (high-low)/(2.*pi)  # [1] (2.3.14) w/ (2.3.7)
    res = np.sqrt(2 * (1 -R))
    
    return res

def maxchange_std (df, startval, lagarray, method='forward'):
    
    d_list=np.zeros((len(lagarray)))
    j=0
    
    def checkforwardbounds(df,startval,lag):
        if (startval+i)>=len(df):
            raise ValueError('Lag value extends beyond length of timeseries')            
    
    def checkbackwardbounds(df,startval,lag):
        if (startval-i)<0:
            raise ValueError('Lag value extends beyond length of timeseries')  

    
    for i in lagarray:
        
        if method=='forward':
            checkforwardbounds(df,startval,i)
            b=df.iloc[startval:startval+i]
            std= mycircstd(b, high=360, low=0) 
            d_list[j]=std
                
        j+=1
    return d_list


def new_temporal_analysis (dir_df, speed_df, lagarray, method='forward'):
    '''
# This function takes a single starting value from a column of directional data in a pandas df or np array
# and returns a list of values at each desired lag.

# Note, directional data must be between 0-360 degrees.
# Need to convert data if in radians and/or if it goes from -180 to 180.


#inputs: df - one column or array of directional data bounded between 0-360
#        startval - first point of wind data that you want to analyze
#        lagarray - array of time lags that you wish to calculate. a lag of 1=.1 sec
#        method - time direction in which lags are calculated 
#        options are forward, backward, average, max, or min. 
#             if no method is selected, forward is the default
    '''
    
    avg_d_list=np.zeros((len(lagarray)))
    avg_s_list=np.zeros((len(lagarray)))
    std_d_list=np.zeros((len(lagarray)))
    std_s_list=np.zeros((len(lagarray)))
    j=0
    
    def checkforwardbounds(dir_df,startval,lag): 
    #compute the standard deviation backwards if the start value+lag will be greater than the length of the df
        if (startval+i)>=len(dir_df): 
            dirchunk=dir_df.iloc[(startval-i):startval]
            speedchunk=speed_df.iloc[(startval-i):startval]
            avg_d_list[j]=scipy.stats.circmean(dirchunk,high=360, low=0)
            avg_s_list[j]= np.mean(speedchunk)
            std_d_list[j]=mycircstd(dirchunk, high=360, low=0) 
            std_s_list[j]=np.std(speedchunk)   
         
    
    def fixangle(angle):
        if angle>=180:
            angle=360-angle
        return angle
    
    
    for i in lagarray:
        
        if method=='forward': #currently only forward - will eventually update
            startval=int(dir_df.sample(1).index.to_numpy()) #get a random start value for each lag
            checkforwardbounds(dir_df,startval,i)
            dirchunk=dir_df.iloc[startval:(startval+i)]
            speedchunk=speed_df.iloc[startval:(startval+i)]
            avg_d_list[j]=scipy.stats.circmean(dirchunk,high=360, low=0)
            avg_s_list[j]= np.mean(speedchunk)
            std_d_list[j]=mycircstd(dirchunk, high=360, low=0) 
            std_s_list[j]=np.std(speedchunk)
            
        j=j+1
    return avg_d_list, avg_s_list, std_d_list, std_s_list