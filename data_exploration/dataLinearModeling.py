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
import scipy.stats as st
from scipy.stats import kurtosis

#classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier

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
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None


# performance
import time
import multiprocessing as mp
dir_save = '../../Figure/container_odor/'
dir = '~/DataAnalysis/data/Sprints/HighRes/'

def load_dataframe():
 
  # dir_save = '../../../Research/Images/container_odor/'
  windy = pd.read_hdf(dir+'Windy/WindyStatsTime_std.h5')
  notwindy = pd.read_hdf(dir+'NotWindy/NotWindyStatsTime_std.h5')
  forest = pd.read_hdf(dir+'Forest/ForestStatsTime_std.h5')
  print('Done Loading Data')
  return windy,notwindy,forest



def get_timed_rows(dataframe,duration_of_encounters):
    x = dataframe.sample(1)
    A = x.mean_time.values.round(0) - duration_of_encounters
    B = x.mean_time.values.round(0)
    timed_rows = dataframe.loc[(dataframe.mean_time > A[0]) & (dataframe.mean_time < B[0])]
#     display(timed_rows)
    return timed_rows
    
def get_timed_encounter_stats(dataframe, distance_class, duration_of_encounters):
    df_q = dataframe.query('type == ' + str(distance_class))   
    df_q.reset_index(inplace=True, drop=True)     
    Nrows = get_timed_rows(df_q,duration_of_encounters)
    avg_dist = np.mean(Nrows.avg_dist_from_source)
    mean_time_whiff=np.mean(Nrows.mean_time)
    pack_data=np.vstack([Nrows.mean_concentration,Nrows.mean_ef,Nrows.log_whiff,Nrows.whiff_ma,Nrows.std_whiff])
    return pack_data,avg_dist,len(Nrows),mean_time_whiff
#     return np.ravel( Nrows[['mean_concentration','mean_ef','log_whiff','mean_ma']].values ),avg_dist

def gather_stat_timed(dataframe, distance_class, duration_of_encounters,X,y,D,N,T):
    for i in range(500):
        xx,dx,n,t=get_timed_encounter_stats(dataframe, distance_class, duration_of_encounters)
        X.append(xx)
        D.append(dx)
        y.append(distance_class)
        N.append(n)
        T.append(t)

    return X,y,D,N,T

def train_test(trainset,lookback_time,stat_to_test):

    D_train=[]
    D_test=[]
    mean_time_train=[]
    mean_time_test=[]
    Xtest = []
    ytest = []
    Xtrain = []
    ytrain = []
    Nrows_train = []
    Nrows_test = []

    for distance_class in [0,1]:
        Xtrain, ytrain, D_train,Nrows_train,mean_time_train = gather_stat_timed(trainset,distance_class,
                        lookback_time, Xtrain,ytrain,D_train,Nrows_train,
                        mean_time_train)


    # for distance_class in [0,1]:
    #     Xtest,ytest,D_test,Nrows_test,mean_time_test = gather_stat_timed(testset,distance_class,
    #                     lookback_time, Xtest,ytest,D_test,Nrows_test,
    #                     mean_time_test)    

    ## Done separately because Forest dataset has two classes
    ## Training set
    cols=['mc_min','mc_max','mc_mean','mc_std_dev','mc_k',
             'wf_min','wf_max','wf_mean','wf_std_dev','wf_k',
             'wd_min','wd_max','wd_mean','wd_std_dev','wd_k',
             'ma_min','ma_max','ma_mean','ma_std_dev','ma_k',
             'st_min','st_max','st_mean','st_std_dev','st_k']

    traindf=pd.DataFrame(columns = cols)
    c1=[]
    for i in range(len(Xtrain)):
        if(np.size(Xtrain[i])==0):
            c1.append(i)
            continue
        else:
            X=[]
            for j in range(len(Xtrain[i])):
                X.append(calc_val(Xtrain[i][j]))
            traindf.loc[i]=np.ravel(X)

    # ## Test set
    # testdf=pd.DataFrame(columns = cols)
    # c2=[]
    # for i in range(len(Xtest)):
    #     if(np.size(Xtest[i])==0):
    #         c2.append(i)
    #         continue
    #     else:
    #         Y=[]
    #         for j in range(len(Xtest[i])):
    #             Y.append(calc_val(Xtest[i][j]))
    #         testdf.loc[i]=np.ravel(Y)

    traindf['distance']=np.delete(D_train, c1)
    traindf['mean_whiff_time'] = np.delete(mean_time_train, c1)
    # testdf['distance']=np.delete(D_test,c2)
    # testdf['mean_whiff_time'] = np.delete(mean_time_test, c2)

    scaler = MinMaxScaler().fit(traindf[traindf.columns])
    traindf[traindf.columns]=scaler.transform(traindf[traindf.columns])
    # testdf[testdf.columns]=scaler.transform(testdf[testdf.columns])


    x = traindf[stat_to_test]
    distance = smf.ols(formula='distance~x',data=traindf).fit()

    return distance.rsquared, distance.aic 


def calc_val(X):
    return np.ravel([np.min(X),np.max(X),np.mean(X),np.std(X),kurtosis(X)])

def bootstrap_anova(inputs):
    trainset,lookback_time,stat_to_test=inputs
    
    aic_list = []
    rsquared_list=[]
    for y in range(0,50):
        rsquared,aic = train_test(trainset,lookback_time,stat_to_test)
        rsquared_list.append(rsquared)
        aic_list.append(aic)

    return rsquared_list, aic_list

def main():
    windy,nwindy,forest=load_dataframe()
    desert = pd.concat([nwindy,windy])
    desert.reset_index(inplace=True, drop=True) 

    column_names=['mc_min','mc_max','mc_mean','mc_std_dev','mc_k',
             'wf_min','wf_max','wf_mean','wf_std_dev','wf_k',
             'wd_min','wd_max','wd_mean','wd_std_dev','wd_k',
             'ma_min','ma_max','ma_mean','ma_std_dev','ma_k',
             'st_min','st_max','st_mean','st_std_dev','st_k']
    lookback_time = 10
    aic_df = pd.DataFrame(columns = column_names)
    rsquared_df= pd.DataFrame(columns = column_names)
     
    inputs = [[forest,lookback_time,x] for x in column_names]
    pool = mp.Pool(processes=(mp.cpu_count()-1))
    rsquared_list,aic_list=zip(*pool.map(bootstrap_anova, inputs ))
    pool.terminate()
    print('Finished Calculating')
    for i in range (len(rsquared_list)):
        rsquared_df.iloc[:,i]=np.ravel(rsquared_list[i])
        aic_df.iloc[:,i]=np.ravel(aic_list[i])

    print('Saving DataFrame')
    rsquared_df.to_hdf(dir+'Forest/Forest_Rsquared.h5', key='rsquared_df', mode='w')
    aic_df.to_hdf(dir+'Forest/Forest_Aic.h5', key='aic_df', mode='w')
      
      

if __name__ == "__main__":
  # execute only if run as a script
  main()