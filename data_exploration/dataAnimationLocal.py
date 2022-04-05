# dataframes
import pandas as pd

# math
pd.TimeSeries = pd.Series 
import numpy as np

# plots
import matplotlib.pyplot as plt
import figurefirst
from figurefirst import FigureLayout,mpl_functions
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar

# performance
import time
import multiprocessing as mp
dir_save = '../../Figure/container_odor/'

def load_dataframe(s):
  dir = '~/Documents/Myfiles/DataAnalysis/data/Sprints/LowRes/'
  # dir_save = '../../../Research/Images/container_odor/'
  df = pd.read_hdf(dir+s)
  print('Done Loading Data')
  return df

def plot_time_series(dataframe):
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

# def streakline_container(eastwest,northsouth):
#   for i in range((len(eastwest))-1,-1, -1): 
#     eastwest = np.resize(np.array([eastwest-df.corrected_u[i]*dt]),(1,i)).flatten() 
#     northsouth = np.resize(np.array([northsouth-df.corrected_v[i]*dt]),(1,i)).flatten()
#     x.loc[i]=np.pad(eastwest, ((len(df)-len(eastwest)),0),'constant', constant_values=(0))
#     y.loc[i]=np.pad(northsouth, ((len(df)-len(northsouth)),0),'constant', constant_values=(0))

# def streakline_calculation(df):
#   et = [np.sum(df.corrected_u[j:])*dt for j in range(0,len(df))]
#   nt = [np.sum(df.corrected_v[j:])*dt for j in range(0,len(df))]
#   return et, nt

# def prepare_df(df):
#   df.pop(df.columns[0])
#   strings=[]
#   for i in range(len(df),0,-1):
#     strings.append("p"+str(i))
#   df.columns=strings
#   df.reset_index(inplace=True, drop=True) 
#   return df

def c_bar(f, ax, var):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    cbar=f.colorbar(var, cax=cax, orientation = 'vertical', ticks=[0, 10])
    cbar.ax.set_yticklabels(['0', '10'])
    cbar.ax.set_ylabel('Odor(V)', rotation=-90, labelpad=-5)

def streakline_encounter_plot(inputs):
  
  i,x,y,df=inputs
  tail=500
  f= plt.figure(figsize =(8,5))
  ax = plt.axes (xlim=(-16,12), ylim=(-5,25))
  ax.plot(0,0, 'x', label="Source")
  area = (np.arange(start =i, stop = 0 , step = -1))**4
  ax.scatter(x.loc[i][x.loc[i]!=0],y.loc[i][y.loc[i]!=0], 
              c='#FFA500', alpha = 0.5, edgecolors='none',
              s=np.sqrt(area),label="Streakline Derived\nfrom Wind")

  if (i>tail):          
    var = ax.scatter(df.xsrc[(i-tail):i],df.ysrc[(i-tail):i], c = df.odor[(i-tail):i], cmap = 'inferno', vmin =0 , vmax = 10, 
                s =12)
    c_bar(f, ax, var)
  else:
    var = ax.scatter(df.xsrc[:i],df.ysrc[:i], c = df.odor[:i],cmap = 'inferno', vmin =0 , vmax = 10, 
                s =12 )       
    c_bar(f, ax, var)

  ax.set_xlabel('Longitude, m')
  ax.set_ylabel('Latitude, m')
  
  ax.set_xticks([-15,0,15])
  ax.set_yticks([-5,10,25])
  
  f.tight_layout(pad=1)
  figurefirst.mpl_functions.set_fontsize(f, 22)
  ax.text(-15.5, -4.2,"time= " + str('{:.4g}'.format(df.time[i])) + " secs", 
            fontsize=12)
  lgnd=ax.legend(edgecolor='black', framealpha=1, bbox_to_anchor=(1,1))
  lgnd.legendHandles[1]._sizes = [35]
  
  f.savefig(dir_save + "plot" + str(i) + ".jpg", dpi=300, bbox_inches = "tight")
  plt.close()


def main():
  ## 2D time series comparison
  x = load_dataframe('HWx1.h5')
  y = load_dataframe('HWy1.h5')
  df = load_dataframe('Windy.h5')
  df = df[df['time'].between(0, 500)]
  # dt = df.master_time[1]-df.master_time[0]
  # et, nt = streakline_calculation(df,dt)
  # ew, ns = streakline_container(et,nt,df,dt)
  # print('Calculated streakline')
  inputs = [[i,x,y,df] for i in range(0,100)]
  pool = mp.Pool(processes=(4))
  pool.map(streakline_encounter_plot, inputs)
  pool.terminate()
  print('\n Finished Plotting Time Series')

if __name__ == "__main__":
  # execute only if run as a script
  main()