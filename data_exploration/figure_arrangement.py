# dataframes
import pandas as pd

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
import multiprocessing as mp

#plots
import figurefirst
from figurefirst import FigureLayout,mpl_functions
import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable # for colorbar
import seaborn as sns
sns.set()

import pylustrator as pyl
# pyl.start()

dir='/Users/an/Documents/MyFiles/DataAnalysis/Figure/'

def methodsFigure01():
  ## TODO: will need to plot all here instead of importing as photos
  ## Setup Image and other stuffs need to be added text in the editor
  ## Arrows need to be added later in PS
  
  img1 = mpimg.imread(dir+'DesertFigure.png')
  img2 = mpimg.imread(dir+'Setup.jpg')
  img3 = mpimg.imread(dir+'/Plots/DesertDirectionScatter.jpeg')
  img4 = mpimg.imread(dir+'/Plots/DesertSpeedScatter.jpeg')  
  img5 = mpimg.imread(dir+'OdorConcentrationNotWindy.jpeg')   
  img6 = mpimg.imread(dir+'DesertSetup.jpg')   
  img7 = mpimg.imread(dir+'IntermittencyBlank.jpg')   

  f,ax=plt.subplots(7,1, figsize=(6.5,4))
  ax[0].imshow(img2)
  ax[1].imshow(img1)
  ax[2].imshow(img3)
  ax[3].imshow(img4)
  ax[4].imshow(img5)
  ax[5].imshow(img6)
  ax[6].imshow(img7)
  for i in range(0,7):
    mpl_functions.adjust_spines(ax[i],'none',
    spine_locations={},smart_bounds=True,linewidth=1)
  
  #% start: automatic generated code from pylustrator
  plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
  # import matplotlib as mpl
  plt.figure(1).axes[0].set_position([0.041187, 0.195866, 0.228420, 0.404432])
  plt.figure(1).axes[1].set_position([0.041367, 0.530385, 0.228420, 0.404432])
  plt.figure(1).axes[2].set_position([0.591852, 0.664148, 0.363416, 0.270669])
  plt.figure(1).axes[3].set_position([0.591852, 0.386592, 0.363416, 0.270669])
  plt.figure(1).axes[4].set_position([0.268942, 0.178598, 0.228420, 0.353171])
  plt.figure(1).axes[5].set_position([0.268942, 0.530385, 0.308202, 0.353171])
  plt.figure(1).axes[6].set_position([0.503937, 0.168307, 0.451048, 0.196850])
  #% end: automatic generated code from pylustrator
  # plt.show()
  # f.savefig(dir+'temp_methods1.jpg')
  f.savefig('../../Figure/temp_methods1.jpeg', dpi=300, bbox_inches = "tight")
def main():
  methodsFigure01()
  
if __name__ == "__main__":
  # execute only if run as a script
  main()