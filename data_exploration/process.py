#user-defined
import ipynb.fs.full.helper as hp

#dataframes
import pandas as pd
import h5py
import datetime as dt

#math
import numpy as np
import math
import scipy.fftpack
from scipy import signal
import scipy.interpolate as interpolate
from scipy.spatial.distance import cdist
from scipy import integrate
pd.TimeSeries = pd.Series 

#gps
from geopy import distance

#plots
import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#others
import time

def load_data():
  set_number = 5
  dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set0'+str(set_number)+'/'
  windbag = 'wind0'+str(set_number)+'Run03.hdf'
  westeast_load = 'ewdata0'+str(set_number)+'Run03.hdf'
  northsouth_load= 'nsdata0'+str(set_number)+'Run03.hdf'
  odor_load = 'Interpolated_'+str(set_number)+'.h5'
  wind_load = 'wind0'+str(set_number)+'Run03_Interpolated.hdf'
  wind_load_small = 'wind0'+str(set_number)+'Run03_Interpolatedsmall.hdf'
  puffsize = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/puff_size.hdf')
  odor= pd.read_hdf(dir+odor_load)
  windn = pd.read_hdf(dir+wind_load)
  windsm = pd.read_hdf(dir+wind_load_small)
  print('Done Loading Data')
  return windn


def get_position(df):
  # summation till Nth particle
  eastwest = [np.sum(df.U[j:])*dt for j in range(0,len(df))]
  northsouth = [np.sum(df.V[j:])*dt for j in range(0,len(df))]
  odor_position = np.array([[df.xsrc[i],df.ysrc[i]] for i in range (len(df))]) # odor particle
  
  return eastwest, northsouth, odor_position

def create_wind_position_frame(windn, odor_position): 
  
  df = pd.DataFrame()
  df = windn
  dt= df.master_time[1]-df.master_time[0]
  odor_presence=[]
  eastwest , northsouth , odor_position = get_position(df)

  l = len(df)
  for i in range((len(eastwest))-1, -1, -1):        # moving backwards
    odor_pos = [odor_position[i]]  
    if(i == 0):
        radius = np.zeros(1)
        wind_pos = np.array([[0,0]])
    else:
        eastwest = np.resize(np.array([eastwest-df.U[i]*dt]),(1,i)).flatten()     # caculating previous step and updating EW
        northsouth = np.resize(np.array([northsouth-df.V[i]*dt]),(1,i)).flatten() # same as EW
        wind_pos = np.vstack([eastwest,northsouth]).T                             # forming 2D pairs
        radius = np.arange(start = i, stop = 0, step = -1)**2*0.01                # radius
        #TODO: Model better radius
        
    distance = cdist(odor_pos,wind_pos).flatten()   # cdist compares distance for all the points in both arrays
    x = distance<=radius.any()                      # checking if any distance at ith time is equal or less than radius
    if (x.any() == True):
        odor_presence.append(1)
    else:
        odor_presence.append(0)

  return odor_presence

def main():
  windn = load_data()
  odor_presence = create_wind_position_frame(windn)
  

if __name__ == "__main__":
  # execute only if run as a script
  main()