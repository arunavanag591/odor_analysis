# dataframes
import pandas as pd
import h5py

# math
pd.TimeSeries = pd.Series 

# plots
import matplotlib.pyplot as plt
import figurefirst 

# performance
import time
import multiprocessing as mp

def load_dataframe(s_number):
  set_number = s_number

  dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/LowRes/Run0'+str(set_number)+'_expected.h5'
  df = pd.DataFrame()
  df = pd.read_hdf(dir)
  print('Done Loading Data')
  return df

def plot_time_series(dataframe):
  dir_save = '../../../Research/Images/container_odor/'
  i, df = dataframe
  fig = plt.figure()
  ax = plt.axes (xlim=(0,300), ylim=(0,2))
  ax.set_xlabel('Time')
  ax.set_ylabel('Odor Concentration')
  ax.plot(df.sync_time[:i],df.odor[:i])
  figurefirst.mpl_functions.set_fontsize(fig, 15)
  fig.savefig(dir_save + "plot" + str(i) + ".jpg")
  # fig.suptitle('Odor Encountered')
  plt.close()
    
def plot_concentration(df):
  # dir_save = '../../../Research/Images/container_odor/'
  
  f1, (ax1,ax2) = plt.subplots(2, 1,figsize=(20,10))
  ax1.scatter(df.sync_time, df.odor, c=df.odor, s=5, cmap='magma')
  ax1.set_ylabel('Concentration')
  ax1.title.set_text('Encountered Particle')
  ax2.scatter(df.sync_time,df.odor_expected, c=df.odor_expected, s=5, cmap='inferno')
  ax2.set_xlabel('Time (secs)')
  ax2.set_ylabel('Concentration')
  ax2.title.set_text('Calculated Particle')

  # f1.suptitle('Odor Oncentration - Radius time**0.5*0.01', fontsize =20)
  figurefirst.mpl_functions.set_fontsize(f1, 15)
  f1.show()
  plt.show()



def main():
  ## 2D time series comparison

  df = load_dataframe(1)     
  inputs = [[i, df] for i in range(len(df))]
  pool = mp.Pool(processes=(mp.cpu_count()-1))
  pool.map(plot_time_series, inputs)
  pool.terminate()
  print('\n Finished Plotting Time Series')


if __name__ == "__main__":
  # execute only if run as a script
  main()