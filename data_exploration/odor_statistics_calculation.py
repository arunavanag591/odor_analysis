# user defined functions
import odor_statistics_lib as osm

# dataframes
import pandas as pd
import h5py

#suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.TimeSeries = pd.Series 

#math
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import signal
from scipy import stats

#plots
import figurefirst
from figurefirst import FigureLayout,mpl_functions
import matplotlib.ticker as mtick
import pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None

# performance
import time
import multiprocessing as mp

#misc
import time
np.set_printoptions(suppress=True)

distance = []
dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/HighRes/'

def load_dataframe():

  df = pd.DataFrame()
  df = pd.read_hdf(dir+'Windy/WindyMASigned.h5')
  print('Done Loading Data')
  return df


def get_statistics(df,index,fdf):
  osm.avg_distance(df,index,fdf)
  osm.mean_conc(df,index,fdf)
  osm.motion_statistics(df,index,fdf)
  osm.whiff_blank_duration(df,index,fdf)
  osm.trajectory_speed(df,index,fdf)
  osm.encounter_frequency(df,index,fdf)
  osm.mean_avg(df,index,fdf)


##TODO: add a new row for every iteration for all the values
def results_summary_to_dataframe(results):
  whiff = []
  ma = []
  encounterFrequency= []
  blanks = []
  rsquared = [] 

  for i in range (len(results)):

    whiff.append(results[i].pvalues.whiffs_resid)
    ma.append(results[i].pvalues.movingavg_resid)
    encounterFrequency.append(results[i].pvalues.encounterfreq_resid)
    blanks.append(results[i].pvalues.blanks_resid)
    rsquared.append(results[i].rsquared)

  results_df = pd.DataFrame({"rsquared":rsquared, 
                              "p_whiff_length":whiff,
                              "p_encounter_frequency":encounterFrequency,
                              "p_moving_avg":ma,
                              "p_blanks":blanks
                              })

  #Reordering...
  results_df = results_df[["rsquared","p_whiff_length","p_encounter_frequency","p_moving_avg","p_blanks"]]
  results_df.to_hdf(dir+'R2NotWindy.h5', key='results_df', mode='w')

def get_rsquared_distribution(inputs):
  
  i, dfx = inputs

  df = pd.DataFrame(dfx.drop(dfx.sample(n=700000).index))
  df.reset_index(inplace=True, drop=True) 
  index = osm.get_index(df)
    
  fdf=pd.DataFrame()

  osm.avg_distance(df,index,fdf)
  osm.motion_statistics(df,index,fdf)
  osm.whiff_blank_duration(df,index,fdf)
  osm.trajectory_speed(df,index,fdf)
  osm.encounter_frequency(df,index,fdf)
  osm.mean_avg(df,index,fdf)
  fdf = osm.sort_by_distance(fdf)

  model = osm.get_distance_statsmodel(fdf)
  
  return model

def get_cutoff_freq_stats_plot(fpval,rsquare,cutoff_freq):
  f,(ax1,ax2)=plt.subplots(2,1,figsize=(8,12))
  ax1.scatter(cutoff_freq, np.log10(fpval))
  ax2.scatter(cutoff_freq, rsquare)
  ax1.grid(False)
  ax2.grid(False)
  mpl_functions.adjust_spines(ax1,['left'],spine_locations={},
                              smart_bounds=True, yticks=[-45,-20,0],linewidth=1)

  mpl_functions.adjust_spines(ax2,['left', 'bottom'],spine_locations={},
                              smart_bounds=True, xticks=[0,15,30,45,60,75,90],yticks=[0.02,0.05,0.08],linewidth=1)

  ax1.yaxis.set_label_coords(-0.15, 0.5)
  ax1.set_yticklabels([r'$10^{-4.5}$',r'$10^{-2}$', r'$10^0$'])
  ax1.set_xlabel('Cutoff Frequency, Hz')
  ax1.set_ylabel('log(F-Prob)')

  ax2.set_xlabel('Cutoff Frequency, Hz')
  ax2.set_ylabel('$R^{2}$')
  f.suptitle("$F-Prob$ & $R^{2}$ HWS\nLPF 0<Cutoff<90")
  f.tight_layout(pad=2)
  figurefirst.mpl_functions.set_fontsize(f, 22)
  f.savefig('../../Figure/LPF_HWSt.jpeg', dpi=300, bbox_inches = "tight")

def get_cutoff_freq_stats(inputs):
  i,cutoff_freq=inputs
  df = load_dataframe() 
  sos = signal.butter(2, cutoff_freq[i], 'low',fs=200, output='sos')
  filtered = signal.sosfilt(sos, df.odor)
  df['filtered_odor']=filtered

  index = osm.get_index_filtered(df)
  fdf = pd.DataFrame()
  np.seterr(divide = 'ignore') 
  get_statistics(df,index,fdf)

  whiff_frequency=smf.ols(formula='mean_ef ~ np.abs(fdf.avg_perpendicular_encounter) + np.abs(fdf.avg_parallel_encounter)', data=fdf).fit()
  whiff_duration=smf.ols(formula='log_whiff~ np.abs(fdf.avg_perpendicular_encounter) + np.abs(fdf.avg_parallel_encounter)', data=fdf).fit()
  moving_avg = smf.ols(formula='mean_ma ~ np.abs(fdf.avg_perpendicular_encounter) + np.abs(fdf.avg_parallel_encounter)', data=fdf).fit()

  fdf['whiff_frequency_resid']=whiff_frequency.resid
  fdf['whiff_duration_resid'] = whiff_duration.resid
  fdf['moving_avg_resid'] = moving_avg.resid

  distance=smf.ols(formula='log_avg_dist_from_source ~ whiff_duration_resid  + whiff_frequency_resid + moving_avg_resid', data=fdf).fit()
  return distance.f_pvalue , distance.rsquared


def main():
  # fpval_=[]
  cutoff_freq=(np.linspace(1,90,20)).astype(int)
  inputs = [[i, cutoff_freq] for i in range(0,len(cutoff_freq))]
  pool = mp.Pool(processes=(mp.cpu_count()-1))
  # fpval_,cfreq_= zip(*pool.map(get_cutoff_freq_stats, inputs))
  fpval_,rsquared_=zip(*pool.map(get_cutoff_freq_stats, inputs))
  pool.terminate()      
 
  # print('\n',fpval_)
  fpval_df = pd.DataFrame()
  fpval_df['fpval']=fpval_
  fpval_df['rsquared']=rsquared_
  fpval_df['cutoff_freq']=cutoff_freq
  fpval_df.to_hdf(dir+'Windy/Fpval_Windy.h5', key='fpval', mode='w')

  # distance.append(output)
  # get_cutoff_freq_stats_plot(fpval_,rsquared_,cutoff_freq)
  # print('Creating DataFrame')
  # results_summary_to_dataframe(distance)
  

if __name__ == "__main__":
  # execute only if run as a script
  main()
