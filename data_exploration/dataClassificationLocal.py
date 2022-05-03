# user defined functions
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

def load_dataframe():
  dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/HighRes/'
  # dir_save = '../../../Research/Images/container_odor/'
  windy = create_class_column(pd.read_hdf(dir+'Windy/WindyStats.h5'))
  nwindy = create_class_column(pd.read_hdf(dir+'NotWindy/NotWindyStats.h5'))
  
  print('Done Loading Data')
  return windy,nwindy

def create_class_column(dataframe):
  dataframe.loc[dataframe.avg_dist_from_source < 5, 'type'] = 0
  dataframe.loc[(dataframe.avg_dist_from_source >= 5)  & (dataframe.avg_dist_from_source < 30), 'type'] = 1
  # dataframe.loc[(dataframe.avg_dist_from_source >= 20) & (dataframe.avg_dist_from_source < 30), 'type'] = 2
  dataframe.loc[dataframe.avg_dist_from_source >= 30, 'type'] = 2
  # dataframe.loc[(dataframe.avg_dist_from_source >= 20) & (dataframe.avg_dist_from_source < 30), 'type'] = 3
  # dataframe.loc[dataframe.avg_dist_from_source >= 30, 'type'] = 4
  return dataframe

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
  print("Naive Bayes Test set Score: ",clf.score(Xtest, ytest))
  # print("Naive Bayes Train set Score: ",clf.score(Xtrain, ytrain))
  print("Number of mislabeled points out of a total %d points : %d"
        % (Xtest.shape[0], (ytest != y_pred).sum()))

# for each collection of data to use for the classifier, get statistics from N encounters
def get_N_random_encounter_stats(dataframe, distance_class, N):
  df_q = dataframe.query('type == ' + str(distance_class))
  Nrows = df_q.sample(N)
  return np.ravel( Nrows[['mean_concentration' ,
                          'mean_ef','log_whiff','mean_ma']].values )

def gather_stat(inputs):
  distance_class,dataframe = inputs
  X=[]
  y=[]
  for i in range(len(dataframe)):
    X.append(get_N_random_encounter_stats(dataframe, distance_class, number_of_encounters))
    y.append(distance_class)
  return X,y

def reshape_array(X,y):
  return np.vstack(X), list(itertools.chain.from_iterable(y)) 

number_of_encounters = 12  # features per feature - global variable x

def main():
  windy,nwindy=load_dataframe()
  cl = [0,1,2]
  input1 = [[distance_class,nwindy] for distance_class in cl]
  input2 = [[distance_class,windy] for distance_class in cl]
  pool = mp.Pool(processes=(len(cl)))
  Xtrain,ytrain=zip(*pool.map(gather_stat, input1))
  Xtest,ytest=zip(*pool.map(gather_stat, input2))
  pool.terminate()
  print('Getting Prediction')
  Xtest,ytest = reshape_array(Xtest,ytest)
  Xtrain,ytrain = reshape_array(Xtrain,ytrain)
  get_prediction(Xtest,ytest, Xtrain, ytrain)
  
  
if __name__ == "__main__":
  # execute only if run as a script
  main()