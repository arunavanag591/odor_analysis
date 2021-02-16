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

def puff_radius(row,col):
  if (row > col):
      return (row - col)
  else:
      return (0)
  
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

def create_frame(pos):
  print('create frame starting')
  windframe=pd.DataFrame(pos,columns=['particle{}'.format(x+1) for x in range(len(pos))])
  print('Frame was created, Rearranging')
  return rearrange_frame(windframe)

def find_particle_position(windn):
  #find wind paritcle position  
  posu=[]
  posv=[]
  for i in range(len(windn.master_time)):
      posu.append(integrate.cumtrapz(windn.U[i:],windn.master_time[i:], axis=0, initial = 0.0)) 
      posv.append(integrate.cumtrapz(windn.V[i:],windn.master_time[i:], axis=0, initial = 0.0))
  
  print('Position arryas generated')
  return posu, posv

def rearrange_frame(windframe):
  dfi = pd.DataFrame()
  dfi['index']=windframe.index
  westeast=windframe.T
  westeast.set_index(dfi.index, inplace=True)
  westeast.columns = ['particle' + str(col) for col in westeast.columns]
  delta=pd.DataFrame()
  delta[0] = westeast.iloc[:,0]
  for i in range(1,len(westeast.columns)):
      delta[i]=westeast['particle' + str(i)].shift(periods=i)

  delta.columns = ['particle' + str(col) for col in delta.columns]
  delta=delta.fillna(0)
  print('Frame has be rearranged and returned for saving')
  #delta.to_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Delta.hdf', key='df2', mode='w')
  return delta


def create_wind_position_frame():
  windn=pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Interpolated.hdf')
  windn_sync_time=windn.master_time-windn.master_time[0]
  windn.insert(1,'sync_time',windn_sync_time)
  
  print('Calculating Particle Position')
  #particle position - > returns a list of arrays
  posu , posv = find_particle_position(windn)

  print('Arranging list of arrays to dataframe')
  #returns a rearranged frame for westeast and northsouth wind (x,y)
  westeast = create_frame(posu)
  #northsouth = create_frame(posv)
  
  print('Rearranged frame received, proceeding to save in data directory')
  #save dataframe
  westeast.to_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Delta.hdf', key='df2', mode='w')


def main():
  load_data()
  create_wind_position_frame()
  # row_size=4004
  # col_size=4004
  # puff_data(row_size,col_size)
  # #odor_expectation_plot()

if __name__ == "__main__":
  # execute only if run as a script
  main()