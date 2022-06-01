# user defined functions
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
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#plots
import string
import figurefirst
from figurefirst import FigureLayout,mpl_functions
import matplotlib.ticker as mtick
import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None


# performance
import time
import multiprocessing as mp
dir_save = '../../Figure/container_odor/'
dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/HighRes/'

def load_dataframe():
 
  # dir_save = '../../../Research/Images/container_odor/'
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

def create_class_column_forest(dataframe):
    dataframe.loc[dataframe.avg_dist_from_source < 5, 'type'] = 0
    dataframe.loc[(dataframe.avg_dist_from_source >= 5)  & (dataframe.avg_dist_from_source < 10), 'type'] = 1
    dataframe.loc[dataframe.avg_dist_from_source >= 10, 'type'] = 2
    return dataframe

def check_length(dataframe, Nrows, nrows, N):
  if (len(Nrows) !=N):
      rowsneeded  = N - len(Nrows) 
      Nrows = Nrows.append(dataframe[(nrows.index-rowsneeded).values[0]:(nrows.index).values[0]])
      Nrows = Nrows.sort_index()
      return Nrows
      # check_length(dataframe,Nrows, nrows, N)
  else:
      return Nrows
  
def get_rows(dataframe, N):
  nrows = dataframe.sample(1)
  Nrows = dataframe[(nrows.index).values[0]:(nrows.index+N).values[0]]
  Nrows = check_length(dataframe,Nrows, nrows, N)
  return Nrows

def stack_arrays(a):
  A = np.full((len(a), max(map(len, a))), np.nan)
  for i, aa in enumerate(a):
    A[i, :len(aa)] = aa
  return A

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


# for each collection of data to use for the classifier, get statistics from N encounters
def get_N_consecutive_encounter_stats(dataframe, distance_class, N):
  df_q = dataframe.query('type == ' + str(distance_class))   
  df_q.reset_index(inplace=True, drop=True)     
  Nrows = get_rows(df_q,N)
  return np.ravel( Nrows[['mean_concentration','mean_ef','log_whiff','mean_ma']].values )


# for each collection of data to use for the classifier, get statistics from N encounters
def get_N_random_encounter_stats(dataframe, distance_class, N):
  df_q = dataframe.query('type == ' + str(distance_class))   
  df_q.reset_index(inplace=True, drop=True)     
  Nrows = df_q.sample(N)
  return np.ravel( Nrows[['mean_concentration','mean_ef','log_whiff','mean_ma']].values )


def gather_stat_random(inputs):
#   print(mp.current_process())
  distance_class,dataframe,number_of_encounters = inputs
  X=[]
  y=[]
  for i in range(200):
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

# def class_population_accuracy(ytest,y_pred):
    
#   cm = confusion_matrix(ytest, y_pred)
#   cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()

#   return cm

def class_population_accuracy(ytest,y_pred):
  cm = confusion_matrix(ytest, y_pred)
  class_acc=[]
  # Calculate the accuracy for each one of our classes
  for idx, cls in enumerate([0,1,2]):
    # True negatives are all the samples that are not our current GT class (not the current row) 
    # and were not predicted as the current class (not the current column)
    tn = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
    # True positives are all the samples of our current GT class that were predicted as such
    tp = cm[idx, idx]
    # The accuracy for the current class is ratio between correct predictions to all predictions
    class_acc.append((tp+tn)/np.sum(cm))
  return (class_acc)

# def main():
#   # print(mp.current_process())
#   windy,nwindy,forest=load_dataframe()
#   desert = pd.concat([nwindy,windy])
#   desert.reset_index(inplace=True, drop=True)

#   list_of_scores = []
#   for i in range(1,51):   # i - number of features
#     cl = [0,1,2]
#     input1 = [[distance_class,desert,i] for distance_class in cl]
#     input2 = [[distance_class,forest,i] for distance_class in [0,1]]
#     pool = mp.Pool(processes=(mp.cpu_count()-1))
#     Xtrain,ytrain=zip(*pool.map(gather_stat_consecutive, input1))
#     Xtest,ytest=zip(*pool.map(gather_stat_consecutive, input2))
#     # print(np.asarray(Xtrain).shape)
#     pool.terminate()
#     Xtest,ytest = reshape_array(Xtest,ytest)
#     Xtrain,ytrain = reshape_array(Xtrain,ytrain)
#     list_of_scores.append(get_prediction(Xtest,ytest, Xtrain, ytrain))
#     print(i)
#   print('saving data')
#   score_df = pd.DataFrame()
#   score_df["encounters"]=np.arange(1,len(list_of_scores)+1,1)
#   score_df["accuracy"] = list_of_scores
#   score_df.to_hdf(dir+'Classifier/Scores_desert_forest.h5', key='score_df', mode='w')


def main():
  windy,nwindy,forest=load_dataframe()
  desert = pd.concat([nwindy,windy])
  desert.reset_index(inplace=True, drop=True) 

  accuracydf=pd.DataFrame()
  bootstrap_length=25
  iterator=1
  for features in range(1,51):
    accuracy = []
    
    for bootstrap in range(0,bootstrap_length):
  
      input1 = [[distance_class,desert,features] for distance_class in [0,1,2]]
      input2 = [[distance_class,forest,features] for distance_class in [0,1]]
      pool = mp.Pool(processes=(mp.cpu_count()-1))
      Xtrain,ytrain=zip(*pool.map(gather_stat_random, input1))
      Xtest,ytest=zip(*pool.map(gather_stat_random, input2))
      pool.terminate()
      Xtest,ytest = reshape_array(Xtest,ytest)
      Xtrain,ytrain = reshape_array(Xtrain,ytrain)
      clf = GaussianNB()
      y_pred = clf.fit(Xtrain,ytrain).predict(Xtest)
      
      
      accuracy.append(class_population_accuracy(ytest,y_pred))
    print('feature:',features,' complete fitting')
    # accuracydf.loc[:,'feature_'+str(iterator)] =np.repeat(features,bootstrap_length)
    accuracydf.loc[:,'class_0'+ str(iterator)]=[item[0] for item in accuracy]
    accuracydf.loc[:,'class_1'+ str(iterator)]=[item[1] for item in accuracy]
    accuracydf.loc[:,'class_2'+ str(iterator)]=[item[2] for item in accuracy]
    iterator+=1

  accuracydf.to_hdf(dir+'Classifier/accuracy_Scores_desert_forest.h5', key='accuracydf', mode='w')
  
  
if __name__ == "__main__":
  # execute only if run as a script
  main()