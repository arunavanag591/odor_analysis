# user defined functions
import odor_statistics_modules as osm

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
from scipy.spatial.distance import cdist
from scipy import signal
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
  df = pd.read_hdf(dir+'NotWindy/NotWindyIF.h5')
  print('Done Loading Data')
  return df

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


def main():

  loaded_df = load_dataframe()     

  inputs = [[i, loaded_df] for i in range(50)]
  pool = mp.Pool(processes=(mp.cpu_count()-1))
  output = pool.map(get_rsquared_distribution, inputs)
  pool.terminate()
  distance.append(output)

  print('Creating DataFrame')
  results_summary_to_dataframe(distance[0])
  


if __name__ == "__main__":
  # execute only if run as a script
  main()