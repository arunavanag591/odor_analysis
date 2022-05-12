# user defined functions
from turtle import distance
from venv import create
import odor_statistics_lib as osm

# dataframes
import pandas as pd
import h5py
import itertools

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

#classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#plots
# import string
# import figurefirst
# from figurefirst import FigureLayout,mpl_functions
# import matplotlib.ticker as mtick
# import pylab as plt
# import matplotlib.pyplot as plt
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                                AutoMinorLocator)
# from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
# import seaborn as sns
# sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None


# performance
import time
import multiprocessing as mp
# dir_save = '../../Figure/container_odor/'
dir = '~/DataAnalysis/data/Sprints/HighRes/'

def load_dataframe():
 
  # dir_save = '../../../Research/Images/container_odor/'
  # windy = create_class_column(pd.read_hdf(dir+'Windy/WindyStats.h5'))
  # nwindy = create_class_column(pd.read_hdf(dir+'NotWindy/NotWindyStats.h5'))
  # forest = create_class_column(pd.read_hdf(dir+'Forest/ForestStats.h5'))
  windy = pd.read_hdf(dir+'Windy/WindyStats.h5')
  nwindy = pd.read_hdf(dir+'NotWindy/NotWindyStats.h5')
  forest = pd.read_hdf(dir+'Forest/ForestStats.h5')
  print('Done Loading Data')
  return windy,nwindy,forest

def create_class_column_log(dataframe):
  dataframe.loc[dataframe.log_avg_dist_from_source_signed < 0.7, 'type'] = 0
  dataframe.loc[(dataframe.log_avg_dist_from_source_signed >= 0.7)  & 
                (dataframe.log_avg_dist_from_source_signed < 1.5), 'type'] = 1
  dataframe.loc[dataframe.log_avg_dist_from_source_signed >= 1.5, 'type'] = 2
  return dataframe
  
def create_class_column(dataframe):
  dataframe.loc[dataframe.avg_dist_from_source < 5, 'type'] = 0
  dataframe.loc[(dataframe.avg_dist_from_source >= 5)  & (dataframe.avg_dist_from_source < 30), 'type'] = 1
  dataframe.loc[dataframe.avg_dist_from_source >= 30, 'type'] = 2

  return dataframe

def stack_arrays(a):
  A = np.full((len(a), max(map(len, a))), np.nan)
  for i, aa in enumerate(a):
    A[i, :len(aa)] = aa
  return A

def get_rows(dataframe, N):
  nrows = dataframe.sample(1)
  Nrows = dataframe[(nrows.index).values[0]:(nrows.index+N).values[0]]
  return Nrows

def get_prediction_same_dataframe(X,y):
  # # Train classifier
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
  gnb = GaussianNB()

  # test classifier
  y_pred = gnb.fit(X_train, y_train).predict(X_test)
  print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))

  print("Naive Bayes score: ",gnb.score(X_test, y_test))


def get_prediction(Xtest,ytest, Xtrain, ytrain):
  clf = GaussianNB()
  y_pred = clf.fit(Xtrain,ytrain).predict(Xtest)
  return clf.score(Xtest,ytest)


# for each collection of data to use for the classifier, get statistics from N consecutive encounters
def get_N_consecutive_encounter_stats(dataframe, distance_class, N):
  df_q = dataframe.query('type == ' + str(distance_class))   
  df_q.reset_index(inplace=True, drop=True)     
  Nrows = get_rows(df_q,N)
  return np.ravel( Nrows[['mean_concentration','mean_ef','log_whiff','mean_ma']].values )

# for each collection of data to use for the classifier, get statistics from N random encounters
def get_N_random_encounter_stats(dataframe, distance_class, N):
  df_q = dataframe.query('type == ' + str(distance_class))
  Nrows = df_q.sample(N)
  return np.ravel( Nrows[['mean_concentration','mean_ef','log_whiff','mean_ma']].values )


def gather_stat_random(inputs):
#   print(mp.current_process())
  distance_class,dataframe,number_of_encounters = inputs
  X=[]
  y=[]
  for i in range(1500):
    X.append(get_N_random_encounter_stats(dataframe, distance_class, number_of_encounters))
    y.append(distance_class)
  return X,y

def gather_stat_consecutive(inputs):
#   print(mp.current_process())
  distance_class,dataframe,number_of_encounters = inputs
  X=[]
  y=[]
  for i in range(1500):
    X.append(get_N_consecutive_encounter_stats(dataframe, distance_class, number_of_encounters))
    y.append(distance_class)
  return X,y

def reshape_array(X,y):
  return np.vstack(X), list(itertools.chain.from_iterable(y)) 

# def reshape_array(X,y):
#   return stack_arrays(X), list(itertools.chain.from_iterable(y)) 

def main():
  # print(mp.current_process())
  windy,nwindy,forest=load_dataframe()
  newtest = pd.concat([nwindy,windy])
  newtest.reset_index(inplace=True, drop=True)

  list_of_scores = []
  for i in range(1,51):   # i - number of features
    cl = [0,1,2]
    input1 = [[distance_class,newtest,i] for distance_class in cl]
    input2 = [[distance_class,forest,i] for distance_class in [0,1]]
    pool = mp.Pool(processes=(mp.cpu_count()-1))
    Xtrain,ytrain=zip(*pool.map(gather_stat_random, input1))
    Xtest,ytest=zip(*pool.map(gather_stat_random, input2))
    # print(np.asarray(Xtrain).shape)
    pool.terminate()
    Xtest,ytest = reshape_array(Xtest,ytest)
    Xtrain,ytrain = reshape_array(Xtrain,ytrain)
    list_of_scores.append(get_prediction(Xtest,ytest, Xtrain, ytrain))
    print(i)
  
  print('saving data')
  score_df = pd.DataFrame()
  score_df["encounters"]=np.arange(1,len(list_of_scores)+1,1)
  score_df["accuracy"] = list_of_scores
  score_df.to_hdf(dir+'AccuracyScoresNB/Random/NotLogged/Scoresf.h5', key='score_df', mode='w')

  
  
if __name__ == "__main__":
  # execute only if run as a script
  main()