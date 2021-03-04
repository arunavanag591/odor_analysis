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
import pylab as pyplt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#others
import time
import decimal

def load_dataframe():
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


def get_position(df, dt):
  # summation till Nth particle
  eastwest = [np.sum(df.U[j:]) * dt for j in range(0,len(df))]
  northsouth = [np.sum(df.V[j:]) * dt for j in range(0,len(df))]
  odor_position = np.array([[df.xsrc[i],df.ysrc[i]] for i in range (len(df))]) # odor particle
  
  return eastwest, northsouth, odor_position

def update_frame(odor_presence, wind_data_frame):
  df = wind_data_frame
  #value is compared with encountered particle and concentration is copied into the existing data frame
  odor_expected = []
  for i in range(len(odor_presence)):
    if(odor_presence[i]==1):
        odor_expected.append(df.odor[i])
    else:
        odor_expected.append(0.6) #as per sensor reading anything below 1.5v
  df['odor_expected'] = odor_expected

  return df

def calculate_expected_encounters(windn): 
  
  df = pd.DataFrame()
  df = windn
  dt = df.master_time[1]-df.master_time[0]
  odor_presence=[]
  print('Getting Encounters')
  eastwest , northsouth , odor_position = get_position(df, dt)

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
  
  print('Finishing Calculating Encounters')

  return odor_presence

def plot_time_series(df):
  f, (ax1,ax2) = plt.subplots(2, 1,figsize=(20,10))
  ax1.plot(df.sync_time, df.odor)
  ax1.set_ylabel('Concentration', fontsize=20)
  ax1.title.set_text('Encountered Particle')
  ax2.plot(df.sync_time,df.odor_expected)
  ax2.set_xlabel('Time (secs)', fontsize=20)
  ax2.set_ylabel('Concentration', fontsize=20)
  ax2.title.set_text('Calculated Particle')

  f.suptitle('Plot - Radius time**0.5*0.01', fontsize =20)
  f.show()
  plt.show()

  

def plot_concentration(df):
  f1, (ax1,ax2) = plt.subplots(2, 1,figsize=(20,10))
  ax1.scatter(df.sync_time, df.odor, c=df.odor, s=5, cmap='magma')
  ax1.set_ylabel('Concentration', fontsize=20)
  ax1.title.set_text('Encountered Particle')
  ax2.scatter(df.sync_time,df.odor_expected, c=df.odor_expected, s=5, cmap='magma')
  ax2.set_xlabel('Time (secs)', fontsize=20)
  ax2.set_ylabel('Concentration', fontsize=20)
  ax2.title.set_text('Calculated Particle')

  f1.suptitle('Odor Oncentration - Radius time**0.5*0.01', fontsize =20)
  f1.show()
  plt.show()

def main():
  windn = load_dataframe()  #load wind data
  print('\nComputing Wind Position')
  odor_presence = calculate_expected_encounters(windn)
  print('\nUpdating Wind Data Frame with Calculated Encounters')
  updated_df = update_frame(odor_presence, windn) 
  print('\nPlot Time Series')
  plot_time_series(updated_df)
  print('\nPlot Concentration')
  plot_concentration(updated_df)

if __name__ == "__main__":
  # execute only if run as a script
  main()