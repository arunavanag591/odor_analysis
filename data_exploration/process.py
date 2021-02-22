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

# def load_data():
#   set_number = 5
#   dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set0'+str(set_number)+'/'
#   windbag = 'wind0'+str(set_number)+'Run03.hdf'
#   westeast_load = 'ewdata0'+str(set_number)+'Run03.hdf'
#   northsouth_load= 'nsdata0'+str(set_number)+'Run03.hdf'
#   odor_load = 'Interpolated_'+str(set_number)+'.h5'
#   puffsize = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/puff_size.hdf')
#   odor= pd.read_hdf(dir+odor_load)
#   print('Done Loading Data')
#   return odor, windbag, westeast_load,northsouth_load, odor_load, puffsize

# def slot_data():
  
#   query = 'sync_time >= 100  and sync_time <=110'
#   dfo = odor.query(query)
#   dfwe = we.query(query)
#   dfns = ns.query(query)
#   geodfsub = geodf.query(query)
#   dfwe = dfwe.reset_index()
#   dfns = dfns.reset_index()
#   geodfsub = geodfsub.reset_index()
#   return (geodfsub)


# def animate_odor_jpeg():  #animation using individual jpegs
#   dir = '../../../Research/Images/container_odor/'
#   N = 0
#   for i in range(len(geodf.xsrc)):
#       fig = plt.figure()
#       fig.suptitle('Odor Encounters')
#       ax = plt.axes (xlim=(-8,15), ylim=(-2,30))
#       ax.set_xlabel('Longitude (meters)')
#       ax.set_ylabel('Latitude(meters)')
#       if (i<=3000):
#           ax.scatter(geodf.xsrc[:i],geodf.ysrc[:i], c=geodf.odor[:i], cmap='magma', s=15)
#       else:
#           ax.scatter(geodf.xsrc[N:i],geodf.ysrc[N:i], c=geodf.odor[N:i], cmap='magma', s=15)
#           N=N+1
          
#       fig.savefig(dir + "plot" + str(i) + ".jpg")
#       plt.close()

# def animate_wind_jpeg():   #animation using individual jpegs
#   dir = '../../../Research/Images/container_wind/'
#   for i in range(len(we)):
#       fig = plt.figure()
#       fig.suptitle('Odor Encounters')
#       ax = plt.axes (xlim=(-8,15), ylim=(-2,30))
#       ax.set_xlabel('Longitude (meters)')
#       ax.set_ylabel('Latitude(meters)')
#       ax.scatter(we.loc[i], ns.loc[i],c ='b', cmap='magma', s=puffsize.loc[i]* 0.1)
#       plt.plot(0,0, marker='x', markersize=15)
#       fig.savefig(dir + "plot" + str(i) + ".jpg")
#       plt.close()

# def puff_radius(row,col):
#   if (row > col):
#       return ((row - col)*0.001)
#   else:
#       return (0)
# def find_particle_position(windn):
#   #find wind paritcle position  
#   # westeast=pd.DataFrame(integrate.cumtrapz(windn.U[0:],windn.master_time[0:], axis=0, initial = 0.0))
#   # northsouth=pd.DataFrame(integrate.cumtrapz(windn.V[0:],windn.master_time[0:], axis=0, initial = 0.0))
  
#   print('generating particles for U')
#   positionU = [integrate.cumtrapz(windn.U[i:],windn.master_time[i:], axis=0, initial = 0.0) for i in range(len(windn.U))]
#   print('generating particles for V')
#   positionV = [integrate.cumtrapz(windn.V[i:],windn.master_time[i:], axis=0, initial = 0.0) for i in range(len(windn.V))]

#   return positionU, positionV

# def wind_particle_position(WE, NS, col,row):
#   if(col == 0):
#       pos_x = WE[col][row]
#       pos_y = NS[col][row]
#       return pos_x , pos_y
  
#   elif(col > 0):
#       if (col > row):
#           pos_x = 0.0
#           pos_y = 0.0
#           return pos_x , pos_y
      
#       else:
#           pos_x = WE[col-col][row-col]
#           pos_y = NS[col-col][row-col]
#           return pos_x, pos_y

# def odor_locations(WE, NS):
#   odor_presence=[]
#   row = 0
#   col = 0
#   l = len(WE)
#   for row in range(l):
#       for col in range(l):
#           k=0
#           if (col > l):
#               row=+1
#           else:
#               if(row == l):
#                   break
#               else:             
#                   windx, windy = wind_particle_position(WE, NS, col,row)
#                   wind_pos = np.array([windx, windy])     
#                   odor_pos = np.array((geodf.xsrc[row],geodf.ysrc[row]))    
#                   distance = np.linalg.norm(wind_pos-odor_pos)

#                   if(distance<=puff_radius(col,row)):
#                       k+=1            
#                   else:
#                       k+=0    
#       if(k>0):
#           odor_presence.append(1)
#       else:
#           odor_presence.append(0)   


#   return odor_presence

# def rearrange_frame(windframe):
  # dfi = pd.DataFrame()
  # dfi['index']=windframe.index
  # westeast=windframe.T
  # westeast.set_index(dfi.index, inplace=True)
  # westeast.columns = ['particle' + str(col) for col in westeast.columns]
  # delta=pd.DataFrame()
  # delta[0] = westeast.iloc[:,0]
  # for i in range(1,len(westeast.columns)):
  #     delta[i]=westeast['particle' + str(i)].shift(periods=i)

  # delta.columns = ['particle' + str(col) for col in delta.columns]
  # delta=delta.fillna(0)
  # print('Frame has be rearranged and returned for saving')
  # #delta.to_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Delta.hdf', key='df2', mode='w')
  # return delta

def get_new_frame(windn , odor_presence):
  odor_expected = []
  for i in range(len(odor_presence)):
      if(odor_presence[i]==1):
          odor_expected.append(windn.odor[i])
      else:
          odor_expected.append(0)
  windn['odor_expected'] = odor_expected
  return windn

def get_particle(windn, l):
  if l == 0:
      return np.array([[0,0]]) 
  else:       
      dt= windn.master_time[1]-windn.master_time[0]
      a = [np.sum(windn.U[i:l])*dt for i in range(l)]
      b = [np.sum(windn.V[i:l])*dt for i in range(l)]
      pos = np.vstack([a,b]).T
      return pos

def get_radius(i):
  if (i == 0):
      return np.zeros(1)
  else:
      a = np.arange(start = i, stop = 0, step = -1)*0.01
#         radius = np.resize(np.insert((a),i,np.zeros(len(delta)-i)),(1,len(delta)))
      return a.flatten()


def create_wind_position_frame():
  windn = pd.read_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_Interpolated.hdf')
  odor_presence = []
  l = len(windn)
  # farthest_odor_point = 10
  print('Computing Encountered to Expected Distance')
  for i in range(l):
    print('time', i)
    odor_pos = np.array([[windn.xsrc[i],windn.ysrc[i]]]) 
    #print(odor_pos)
    wind_pos = np.array(get_particle(windn, i))
    #print(wind_pos)
    distance = cdist(odor_pos,wind_pos).flatten()   
    #print(distance)
    x = distance<=get_radius(i).any()
    if (x.any() == True):
        odor_presence.append(1)
    else:
        odor_presence.append(0)

  print('creating new frame')
  wind = get_new_frame(windn, odor_presence)
  print('Saving DataFrame')
  wind.to_hdf('~/Documents/Myfiles/DataAnalysis/data/Sprints/Run03/Set05/wind05Run03_ExpectedOdor.hdf', key='df2', mode='w')



def main():
  #load_data()
  create_wind_position_frame()
  

if __name__ == "__main__":
  # execute only if run as a script
  main()