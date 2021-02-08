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
from scipy import integrate
pd.TimeSeries = pd.Series 

#gps
from geopy import distance

#plots
import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_data():
  set_number = 5
  dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set0'+str(set_number)+'/'
  windbag = 'wind0'+str(set_number)+'Run03.hdf'
  westeast_load = 'ewdata0'+str(set_number)+'Run03.hdf'
  northsouth_load= 'nsdata0'+str(set_number)+'Run03.hdf'
  odor_load = 'Interpolated_'+str(set_number)+'.h5'
  puffsize = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/puff_size.hdf')
  odor= pd.read_hdf(dir+odor_load)
  print('Done Loading Data')
  return odor, windbag, westeast_load,northsouth_load, odor_load, puffsize

def slot_data():
  
  query = 'sync_time >= 100  and sync_time <=110'
  dfo = odor.query(query)
  dfwe = we.query(query)
  dfns = ns.query(query)
  geodfsub = geodf.query(query)
  dfwe = dfwe.reset_index()
  dfns = dfns.reset_index()
  geodfsub = geodfsub.reset_index()
  return (geodfsub)


def animate_odor_jpeg():  #animation using individual jpegs
  dir = '../../../Research/Images/container_odor/'
  N = 0
  for i in range(len(geodf.xsrc)):
      fig = plt.figure()
      fig.suptitle('Odor Encounters')
      ax = plt.axes (xlim=(-8,15), ylim=(-2,30))
      ax.set_xlabel('Longitude (meters)')
      ax.set_ylabel('Latitude(meters)')
      if (i<=3000):
          ax.scatter(geodf.xsrc[:i],geodf.ysrc[:i], c=geodf.odor[:i], cmap='magma', s=15)
      else:
          ax.scatter(geodf.xsrc[N:i],geodf.ysrc[N:i], c=geodf.odor[N:i], cmap='magma', s=15)
          N=N+1
          
      fig.savefig(dir + "plot" + str(i) + ".jpg")
      plt.close()

def animate_wind_jpeg():   #animation using individual jpegs
  dir = '../../../Research/Images/container_wind/'
  for i in range(len(we)):
      fig = plt.figure()
      fig.suptitle('Odor Encounters')
      ax = plt.axes (xlim=(-8,15), ylim=(-2,30))
      ax.set_xlabel('Longitude (meters)')
      ax.set_ylabel('Latitude(meters)')
      ax.scatter(we.loc[i], ns.loc[i],c ='b', cmap='magma', s=puffsize.loc[i]* 0.1)
      plt.plot(0,0, marker='x', markersize=15)
      fig.savefig(dir + "plot" + str(i) + ".jpg")
      plt.close()

def puff_data(rowsize, colsize):
  # creating puff dataframe
  # going to copy to a new frame to keep this container separate as the for loop takes significant time to execute

  puff = pd.DataFrame(index=range(rowsize),columns=range(colsize))
  puffsize=pd.DataFrame()

  for i in range(0, len(puff.columns)):
      puff[i]=puff.index

  puff.columns = ['particle' + str(col) for col in puff.columns] #renaming for looping ease
  puffsize[0] = puff.iloc[:,0] #copying the first columns of data

  #shifting every column by the column-th number
  for i in range(1, len(puff.columns)):
      puffsize[i]=puff['particle' + str(i)].shift(periods=i)

  puffsize=puffsize.fillna(0) #replacing NaN with zeroes
  puffsize.columns = ['particle' + str(col) for col in puffsize.columns] #renaming columns names optional
  puffsize= puffsize.astype(int)
  puffsize.to_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/puff_size.hdf', key='puffsize', mode='w')
  print('Finished generating Dataframe')
  
def odor_expectation_plot():
  ## finding encountered odor withing the calculated odor radius
  odor_presence=[]
  delta = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Gamma.hdf')
  gamma = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Gamma.hdf')
  slot_data()

  # for i in range(len(puffsize)):
  #   k=0
  #   windx=delta.loc[i]
  #   windy=gamma.loc[i]
  #   puff = puffsize.loc[i]
  #   point1 = np.array((geodfsub.xsrc[i],geodfsub.ysrc[i]))
    
  #   for j in range(len(windx)):
  #       point2 = np.array((windx[j],windy[j]))
  #       distance = np.linalg.norm(point1-point2)
        
  #       for x in range(len(puff)):
  #           #puff radius comparison
  #           if(distance<=puff[x]):
  #               k+=1
  #               break
  #           else:
  #               k+=0
  #   if(k>0):
  #       odor_presence.append(1)
  #   else:
  #       odor_presence.append(0)

  #   odor_expected = []
  #   for i in range(len(puffsize)):
  #       if(odor_presence[i]==1):
  #           odor_expected.append(geodfsub.odor[i])
  #       else:
  #           odor_expected.append(0)

def main():
  load_data()
  
  row_size=4004
  col_size=4004
  puff_data(row_size,col_size)
  #odor_expectation_plot()

if __name__ == "__main__":
  # execute only if run as a script
  main()