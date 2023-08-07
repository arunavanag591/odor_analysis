# dataframes
import pandas as pd

#suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.TimeSeries = pd.Series 

from itertools import groupby
from operator import itemgetter

#math
import numpy as np
from scipy.spatial.distance import cdist
from scipy import signal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import multiprocessing as mp

#plots
import figurefirst
import pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
import seaborn as sns
sns.set()

#misc
import time
np.set_printoptions(suppress=True)

def wrapped_angular_diff(a, b):
  return np.arctan2(np.sin(a-b), np.cos(a-b))

def get_index_simulated(df):
  
  idx = []
  df.odorsim = df.odorsim*10
  for i in range(len(df.odorsim)):
      if (df.odorsim[i]>1):
          idx.append(df.index[i])


  index = []
  for k, g in groupby(enumerate(idx),lambda ix : ix[0] - ix[1]):
      index.append((list((map(itemgetter(1), g)))))
  return index


def get_index_filtered(df):
  thres=4.5
  idx = []
  for i in range(len(df.filtered_odor)):
      if (df.filtered_odor[i]>thres):
          idx.append(df.index[i])

  index = [ ]
  for k, g in groupby(enumerate(idx),lambda ix : ix[0] - ix[1]):
      index.append((list((map(itemgetter(1), g)))))

  return index

def get_index(df):
  
  idx = []
  for i in range(len(df.odor)):
      if (df.odor[i]>4.5):
          idx.append(df.index[i])
  index = []
  for k, g in groupby(enumerate(idx),lambda ix : ix[0] - ix[1]):
      index.append((list((map(itemgetter(1), g)))))
  return index

def get_index_forest(df):
  
  idx = []
  for i in range(len(df.odor)):
      if (df.odor[i]>5):
          idx.append(df.index[i])

  index = []
  for k, g in groupby(enumerate(idx),lambda ix : ix[0] - ix[1]):
      index.append((list((map(itemgetter(1), g)))))
  return index

def avg_distance(df,index,fdf): #input ; location ; storage
  
  #Distance
  i = 0
  avg_dist_source = []
  while i<len(index):
    avg_dist_source.append(np.mean(df.distance_from_source[index[i]])) ## _ is declination corrected distance
    i+=1
  fdf['avg_dist_from_source']=avg_dist_source
  fdf['log_avg_dist_from_source']= np.log10(fdf.avg_dist_from_source)

  avg_xsign = []
  i = 0
  while i<len(index):
    avg_xsign.append(np.mean(df.xsign[index[i]]))
    i+=1
  fdf['avg_xsign']=np.sign(avg_xsign)
  fdf['avg_xsign']=fdf['avg_xsign'].replace(0,1)


  avg_dist_from_streakline = []
  i = 0
  while i<len(index):
    avg_dist_from_streakline.append(np.mean(df.nearest_from_streakline[index[i]]))
    i+=1
  fdf['avg_dist_from_streakline']=avg_dist_from_streakline
  fdf['log_avg_dist_from_source_signed'] = fdf.log_avg_dist_from_source*fdf.avg_xsign

def motion_statistics(df,index,fdf):
  
  # RELATIVE MOTION

  rel_parallel_enc = []
  i= 0
  while i<len(index):
    rel_parallel_enc.append(np.mean(df.relative_parallel_comp[index[i]]))
    i+=1
  fdf['avg_parallel_encounter']=rel_parallel_enc

  rel_perpendicular_enc = []
  i= 0
  while i<len(index):
    rel_perpendicular_enc.append(np.mean(df.relative_perpendicular_comp[index[i]]))
    i+=1
  fdf['avg_perpendicular_encounter']=rel_perpendicular_enc

  rel_parallel_inter = []
  i= 0
  while i < len(index):
    if i < (len(index)-1):
      rel_parallel_inter.append(np.mean(df.relative_parallel_comp[index[i][-1]:index[i+1][0]]))
      i+=1
    else:
      rel_parallel_inter.append(0)
      i+=1

  fdf['avg_parallel_intermittency']=rel_parallel_inter

  rel_perpendicular_inter = []
  i= 0
  while i < len(index):
    if i < (len(index)-1):
      rel_perpendicular_inter.append(np.mean(df.relative_perpendicular_comp[index[i][-1]:index[i+1][0]]))
      i+=1
    else:
      rel_perpendicular_inter.append(0)
      i+=1
        
  fdf['avg_perpendicular_intermittency']=rel_perpendicular_inter

def whiff_blank_duration(df,index,fdf):
  
  # time of the encounters
  i = 0
  length_of_encounter = []
  dt = df.time[1]-df.time[0]   ## dt is constant, dt * length gives length of time
  while i < len(index):
    length_of_encounter.append(dt*(len(index[i])))
    i+=1
  fdf['length_of_encounter'] = length_of_encounter

  #time between the encounters
  i = 0
  intermittency = []
  while i < len(index):
    if i < (len(index)-1):
        intermittency.append((index[i+1][0] - index[i][-1])*dt)
        i+=1
    else:
      intermittency.append(0)
      i+=1
  fdf['odor_intermittency'] = intermittency
  fdf['log_whiff']=np.log10(fdf.length_of_encounter)
  fdf['log_blank']=np.log10(fdf.odor_intermittency)


def trajectory_speed(df,index,fdf):
  
  ## Trajectory speed during Intermittency
  i = 0
  speed_at_intermittency=[]
  while i < len(index):
    if i < (len(index)-1):
      x = np.mean(df.gps_linear_x[index[i][-1]:index[i+1][0]])
      y = np.mean(df.gps_linear_y[index[i][-1]:index[i+1][0]])
      z = np.mean(df.gps_linear_z[index[i][-1]:index[i+1][0]])
      speed_at_intermittency.append(np.sqrt(x**2+y**2+z**2))
      i+=1
    else:
      speed_at_intermittency.append(0)
      i+=1

  fdf['speed_at_intermittency'] = speed_at_intermittency

  ## Trajectory speed during Encounters
  i = 0
  speed_at_encounter=[]
  while i < len(index):
    x = np.mean(df.gps_linear_x[index[i]])
    y = np.mean(df.gps_linear_y[index[i]])
    z = np.mean(df.gps_linear_z[index[i]])
    speed_at_encounter.append(np.sqrt(x**2+y**2+z**2))
    i+=1
  fdf['speed_at_encounter'] = speed_at_encounter

## Encounter Frequency calculation 
def encounter_frequency(df,index,fdf,kernel_size,tau): 
  # binary vector
  start = []
  for i in range (len(index)):
      start.append(index[i][0])
  df['efreq'] = np.zeros(len(df))
  df.efreq.iloc[start] = 1

  ## encounter frequency
  def exp_ker(t, tau):
    return np.exp(-t/tau)/tau

  dt = df.time[1]-df.time[0]
  t = np.arange(0,kernel_size,dt)
  # t=df.time[:10]
  kernel = exp_ker(t,tau)

  filtered = signal.convolve(df.efreq, kernel, mode='full', method='auto')
  filtered = filtered[:-(len(t)-1)]
  df['encounter_frequency']=filtered

  #Average Encounter Frequency
  i = 0
  wfreq = []
  while i<len(index):
      wfreq.append(np.mean(df.encounter_frequency[index[i]]))
      i+=1
  fdf['mean_ef'] = wfreq
  return wfreq

  
def std_whiff(df,index,fdf):

  i = 0
  std_whiff = []
  while i<len(index):
      std_whiff.append(np.std(df.odor[index[i]])) 
      i+=1
  fdf['std_whiff']=std_whiff

  
def mean_conc(df,index,fdf):
  #Distance
  i = 0
  mean_concentration = []
  while i<len(index):
      mean_concentration.append(np.mean(df.odor[index[i]])) 
      i+=1
  fdf['mean_concentration']=mean_concentration

def ma_fraction(df,window_size):
  #Mean Average fraction for Whiff MA calculation

  slider = window_size # 200 rows equals 1 second
  window = np.lib.stride_tricks.sliding_window_view(df.index,slider)
  ifact=[]
  for i in range(len(window)):
      ifact.append(np.count_nonzero(np.where(df.odor[window[i]]>4.5))/len(window[i]))

  lst = [0] * (len(df)-len(np.lib.stride_tricks.sliding_window_view(df.index,slider)))
  x = ifact + lst
  df['ma_fraction'] = x

# def mean_avg(df,index,fdf):
#   #Average Intermittency Factor
#   i = 0
#   ifr = []
#   dt = df.time[1]-df.time[0]
#   while i<len(index):
#       ifr.append(np.mean(df.ma_fraction[index[i]]))
#       i+=1
#   fdf['mean_ma'] = ifr

def mean_avg(df,index,fdf):
  indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=200)
  ma = df.odor.rolling(window=indexer, min_periods=1).mean()
  df['ma_inter']=ma
  moving_avg = []
  i=0
  while i<len(index):
      moving_avg.append(np.mean(ma[index[i]])) 
      i+=1
  fdf['whiff_ma']=moving_avg

def mean_t(df,index,fdf):
  i = 0
  avg_time = []
  while i<len(index):
      avg_time.append(np.mean(df.time[index[i]])) 
      i+=1
  fdf['mean_time']=avg_time

def wind_speed(df,index,fdf):
  
  ### Wind speed during encounter and Intermittency
  i = 0
  wind_speed_encounter = []
  while i<len(index):
    wind_speed_encounter.append(np.mean(df.S2[index[i]]))
    i+=1
  fdf['wind_speed_encounter'] = wind_speed_encounter

  ### Wind speed during intermittency
  i = 0
  wind_speed_intermittency = []
  while i<len(index):
    if i < (len(index)-1):
        wind_speed_intermittency.append(np.mean(df.S2[index[i][-1]:index[i+1][0]]))
        i+=1
    else:
        wind_speed_intermittency.append(0)
        i+=1
  fdf['wind_speed_intermittency'] = wind_speed_intermittency

def sort_by_distance(fdf):
  fdf = fdf.sort_values(by=['avg_dist_from_source'])
  fdf.reset_index(inplace=True, drop=True) 

  a = np.array(np.where(fdf.avg_dist_from_source <=10))
  b = np.array(np.where((fdf.avg_dist_from_source > 10) & (fdf.avg_dist_from_source <=30)))
  c = np.array(np.where(fdf.avg_dist_from_source > 30))
  fdf1 = pd.DataFrame()
  fdf2 = pd.DataFrame()
  fdf3 = pd.DataFrame()

  fdf1['distance_from_source_bin'] = np.repeat('0-10(m)',a.flatten().size)
  fdf2['distance_from_source_bin'] = np.repeat('10-30(m)',b.flatten().size)
  fdf3['distance_from_source_bin'] = np.repeat('>30(m)',c.flatten().size)
  fdf['distance_from_source_bin'] = pd.concat([fdf1,fdf2,fdf3], ignore_index=True)
  # 
  p1 = [0]*(a.flatten().size)
  p2 = [1]*(b.flatten().size)
  p3 = [2]*(c.flatten().size)
  p = p1+p2+p3
  fdf['bins_distance']=p

  return fdf
  
def sort_by_distance_forest(fdf):
  fdf = fdf.sort_values(by=['avg_dist_from_source'])
  fdf.reset_index(inplace=True, drop=True) 

  a = np.array(np.where(fdf.avg_dist_from_source <=10))
  b = np.array(np.where((fdf.avg_dist_from_source > 10) & (fdf.avg_dist_from_source <=20)))
  c = np.array(np.where(fdf.avg_dist_from_source > 20))

  fdf1 = pd.DataFrame()
  fdf2 = pd.DataFrame()
  fdf3 = pd.DataFrame()

  fdf1['distance_from_source_bin'] = np.repeat('0-10(m)',a.flatten().size)
  fdf2['distance_from_source_bin'] = np.repeat('10-20(m)',b.flatten().size)
  fdf3['distance_from_source_bin'] = np.repeat('>20(m)',c.flatten().size)
  
  fdf['distance_from_source_bin'] = pd.concat([fdf1,fdf2,fdf3], ignore_index=True)
  # 
  p1 = [0]*(a.flatten().size)
  p2 = [1]*(b.flatten().size)
  p3 = [2]*(c.flatten().size)

  p = p1+p2+p3
  fdf['bins_distance']=p

  return fdf

def get_distance_statsmodel(fdf):
  pd.set_option('use_inf_as_na', True) ## for excluding negative infinity and NaN values 
  encounter_freq=smf.ols(formula='np.log10(mean_encounter_frequency) ~ np.abs(fdf.avg_perpendicular_encounter) + np.abs(fdf.avg_parallel_encounter)', data=fdf).fit()
  whiffs=smf.ols(formula='log_whiff~ np.abs(fdf.avg_perpendicular_encounter) + np.abs(fdf.avg_parallel_encounter)', data=fdf).fit()
  blanks=smf.ols(formula='log_blank ~ np.abs(fdf.avg_perpendicular_intermittency) + np.abs(fdf.avg_parallel_intermittency)', data=fdf).fit()
  movingavg = smf.ols(formula='mean_ma ~ np.abs(fdf.avg_perpendicular_intermittency) + np.abs(fdf.avg_parallel_intermittency)', data=fdf).fit()

  fdf['encounterfreq_resid']=encounter_freq.resid
  fdf['whiffs_resid'] = whiffs.resid
  fdf['blanks_resid'] = blanks.resid
  fdf['movingavg_resid'] = movingavg.resid

  distance=smf.ols(formula='avg_dist_from_source ~ whiffs_resid + movingavg_resid+ encounterfreq_resid + blanks_resid', data=fdf).fit()
  
  return distance


