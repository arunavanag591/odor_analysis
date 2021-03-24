#user-defined
# import ipynb.fs.full.helper as hp

#dataframes
import pandas as pd
import h5py
import datetime as dt

#math
import numpy as np
import math as m
from scipy.spatial.distance import cdist
pd.TimeSeries = pd.Series 

#plots
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#performance
import time
import multiprocessing as mp


def load_dataframe():
  set_number = 5

  dir = '~/dataAnalysis/Sprints/Run03/Set0'+str(set_number)+'/'
  
  wind_load= 'wind0'+str(set_number)+'Run03_Expected_Full.hdf'
  wind_load_small = 'wind0'+str(set_number)+'Run03_Expected_Small.hdf' ## bag saved from datavisoptimization 
                                                                ## with expected odor information

  windn = pd.read_hdf(dir+wind_load)
  windsm = pd.read_hdf(dir+wind_load_small)
  print('Done Loading Data')
  return windn, windsm

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
        odor_expected.append(1)
    else:
        odor_expected.append(0) 
  df['odor_expected'] = odor_expected

  return df

def calculate_expected_encounters(wind): 
  
  #start = time.time()
  df = pd.DataFrame()
  df = wind
  dt = df.master_time[1]-df.master_time[0]
  # odor_presence.clear()
  # odor_presence = []
  print('Getting Encounters')
  eastwest , northsouth , odor_position = get_position(df, dt)

  for i in range((len(eastwest))-1, -1, -1):        # moving backwards
    odor_pos = [odor_position[i]]  
    if(i == 0):
        radius = np.zeros(1)
        wind_pos = np.array([[0,0]])
    else:
        eastwest = np.resize(np.array([eastwest-df.U[i]*dt]),(1,i)).flatten()     # caculating previous step and updating EW
        northsouth = np.resize(np.array([northsouth-df.V[i]*dt]),(1,i)).flatten() # resize is necessary to avoid negative data padding
        wind_pos = np.vstack([eastwest,northsouth]).T                             # forming 2D pairs
        radius = np.arange(start = i, stop = 0, step = -1)**0.5*0.01              # radius
        #TODO: Model better radius
        
    #TODO: Model better radius
    #max_radius= np.max(radius)
    distance = cdist(odor_pos,wind_pos).flatten()          # cdist compares distance 
                                                           # for all the points in both arrays
    
    #distance = distance[distance(distance<max_radius)]    #this step can reduce computation but arises issues 
                                                           #for different length arrays for distance and radius
    
    ## TODO: Find a way to reduce distance array size and compare without increasing the overall execution time
    
    ## NOTE : COMPARING EVERY DISTANCE TO THE CORRESPONDING RADIUS TO SEE IF THE DISTANCE IS LESSER THAN 
    ## THE RADIUS WHICH WOULD INFER THE PARTICLE POSITIONS MATCH APPROXIMATELY AT TIME t. 
    ## OTHERWISE POSITIONS DONT MATCH FOR TIME t.
    
    ## comparing element to element, i.e. radius to its corresponding distance
    x = np.any(distance<=radius)             # generates a boolean values 
                                             # this point can be used later to 
                                             # to check locations and see overlapping as well.

    if x==True:
        # odor_presence.append(1)
        q.put(1)
    else:
        # odor_presence.append(0)
        q.put(0)
  ## flip containers because above iteration is done in reverse order
  # odor_presence = odor_presence[::-1]

  print('Finishing Calculating Encounters')
  #print('Execution time', time.time()-start)
  # return odor_presence

def plot_time_series(input):
  dir_save = '/home/ecc/dataAnalysis/Images/'
  i, df = input
  fig = plt.figure()
  ax = plt.axes (xlim=(0,300))
  ax.set_xlabel('Time')
  ax.set_ylabel('Odor Concentration')
  ax.plot(df.sync_time[:i],df.odor[:i])
  fig.savefig(dir_save + "plot" + str(i) + ".jpg")
  # fig.suptitle('Odor Encountered')
  plt.close()
    
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

def path_animation(windef, windes):
  df = pd.DataFrame()
  df = windes
  dt = df.master_time[1]-df.master_time[0]

  dir_save = '../../../Research/Images/container_odor/'
  count = 0
  N=2500

  print('Getting Encounters')
  eastwest , northsouth , odor_position = get_position(df, dt)
  
  for i in range((len(eastwest))-1,0, -1):   
    fig = plt.figure()
    fig.suptitle('Radius time**0.5*0.01 - Run03_Set05_Small', fontsize =14)
    
    ax = plt.axes (xlim=(-8,15), ylim=(-2,30))            # need to change for all data sets based on limits
    ax.set_xlabel('Longitude (meters)')
    ax.set_ylabel('Latitude(meters)')
    
    #TODO: Ignoring 0th point for now, need to include letter
    
    eastwest = np.resize(np.array([eastwest-df.U[i]*dt]),(1,i)).flatten() 
    northsouth = np.resize(np.array([northsouth-df.V[i]*dt]),(1,i)).flatten()

    area = np.arange(start = i, stop = 0, step = -1)**2*0.04*m.pi #area
    ax.scatter(eastwest, northsouth, c='#FFA500', alpha = 0.3, s=(area/4)) 
    #TODO: find a better relation for s  

    if (count<2499):          
        ax.scatter(df.xsrc[N:i],df.ysrc[N:i], c = df.odor[N:i], cmap = 'inferno', vmin =0 , vmax = 13, s =12)
        N=N-1
        
    else:
        ax.scatter(df.xsrc[:i],df.ysrc[:i], c = df.odor[:i],cmap = 'inferno', vmin =0 , vmax = 13, s =12 )

    count+=1
    
    fig.savefig(dir_save + "plot" + str(i) + ".jpg")
    plt.close()
  
  print('completed generating all figures')

def time_series_animation(windef, windes):
  df = pd.DataFrame(windes)
  dir_save = '../../../Research/Images/container_odor/'
  for i in range((len(df))-1, 0, -1):                     # doing backward or forward does not matter
    fig = plt.figure()
    #fig.suptitle('Odor Encounters')
    ax = plt.axes (xlim=(0,300), ylim=(0,2))              # need to change for all data sets based on limits
    ax.set_xlabel('Time')
    ax.set_ylabel('Odor Concentration')

    ax.plot(df.sync_time[:i],df.odor_expected[:i])
    fig.savefig(dir_save + "plot" + str(i) + ".jpg")
    plt.close()



def main():
  ## 2D time series comparison

  windn ,windsm = load_dataframe()         #load wind data
  # print('\nComputing Wind Position')
  

  # print(odor_presence)
  # print('\nUpdating Wind Data Frame with Calculated Encounters')
  # updated_df = update_frame(odor_presence, windn) 

  print('\nPlot Time Series')
  #multiprocessing
  print('\n DataFrame Length = ', len(windn))
  inputs = [[i, windn] for i in range(len(windn))]
  pool = mp.Pool(processes=(mp.cpu_count()-1))
  pool.map(plot_time_series, inputs)
  pool.terminate()
  print('\n Finished Plotting Time Series')
  # print('\nPlot Concentration')
  # plot_concentration(updated_df)

  ## 2D video comparison 

  # print('\nGenerating plots for path animation')
  # path_animation(windef,windes)
  # print('\nGenerating plots for time series animation')




if __name__ == "__main__":
  # execute only if run as a script
  main()