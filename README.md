# Odor Analysis
This repository consist of the data analysis done for Odor Tracking experiment. 

Preprint: ["Odor source location can be predicted from a time-history of odor statistics for a large-scale outdoor plume"](https://www.biorxiv.org/content/10.1101/2023.07.20.549973v1)



## Figures

Data will be available in data dryad to run the results in the paper. Below are interactive notebooks, which can be used using Jupyter Notebook and run using python 3.8 and inskcape to generate the figures and results. These figures were generated using [figurefirst](https://github.com/FlyRanch/figurefirst) 

#### Main Text Figures: 
1. [Figure 1](/data_exploration/figure/method1.ipynb) : Desert Setup and data description
2. [Figure 2](/data_exploration/figure/method2.ipynb) : Forest Setup 
3. [Figure 3](/data_exploration/figure/streaklinemappingRevised.ipynb) : Mapping of wind data to distance from source coordinates 
4. [Figure 4](/data_exploration/figure/statCalFigure.ipynb) : Odor Statistics calculation
5. [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb) : Mean Meta odor statistics vs Distance from source 
6. [Figure 6](/data_exploration/figure/figureClustering.ipynb) : Heirarchical clustering - relationship between whiff statistcs
7. [Figure 7](/data_exploration/figure/klmfigure.ipynb) : Integrating whiff statistics with relative motion in a Kalman smoother 
8. [Figure 8](/data_exploration/figure/lowpassfilter.ipynb) : Low pass filtering odor signal degrades correlations between whiff statistics and source
distance


#### Supplemental Figure

1. [Figure S1](/data_exploration/figure/Supplemental/verticalMovement.ipynb) : Vertical ambient wind analysis
2. [Figure S2](/data_exploration/figure/Supplemental/motionAnalysis.ipynb) : Agent search motion analysis
3. [Figure S3](/data_exploration/figure/Supplemental/NormalityAnalysis.ipynb) : Normality analysis of the datasets
4. [Figure S4](/data_exploration/figure/Supplemental/windlagfigure.ipynb) : Range of wind characteristics across our datasets 
5. [Figure S5](/data_exploration/figure/Supplemental/timeSpent.ipynb) : Time spent - vs number of encounters
6. [Figure S6](/data_exploration/figure/Supplemental/whiffStatisticsIndividualDatasets.ipynb) : Individual whiff statistics without a lookback time 
7. [Figure S7](/data_exploration/figure/Supplemental/mc_wsd.ipynb) : Mean whiff concentration vs mean whiff standard deviation
8. [Figure S8](/data_exploration/figure/Supplemental/figureAicR2layout.ipynb) : Bootstrapped R-squared and AIC analysis
9. [Figure S9](/data_exploration/figure/Supplemental/windAicParamsAnalysis.ipynb) : AIC filtered coefficiecnts across various wind scenarios
10. [Figure S10](/data_exploration/figure/Supplemental/klmsupplemental.ipynb) : Kalman smoothed estimates of the distance to the odor source zoomed



## Setup Environment

Install <a href = "https://docs.python-guide.org/dev/virtualenvs/"> Virtualenv </a>: ```pip install virtualenv```<br/>

Install Anaconda from <a href = "https://docs.anaconda.com/anaconda/install/linux/">here. </a>



1. Create a Conda Environment:  

   ```bash
   conda create -n FlyDataAnalysis python=3.6  
   ```
2. Create the virtualenv:

    ```
   virtualenv -p /usr/bin/python3.6 dataEnv  
   ```
  
3. Install Packages:

   ```
   pip install pandas
   pip install h5py
   pip install numpy
   pip install matplotlib
   pip install figurefirst
   pip install tables
   conda install --channel conda-forge cartopy  
   ``` 

4. To Install Jupyter Dark Theme (optional - better not to):

   ```bash
   conda install -c conda-forge jupyterthemes 
   ```

### Bash_Aliases Setup
```bash
alias venv="source dataEnv/bin/activate"
alias denv="deactivate"
alias condaenv="conda activate FlyDataAnalysis"
alias dconda="conda deactivate"
alias jread="jupyter notebook"
alias rosenv="source rosenv/bin/activate"
alias start="venv && condaenv"
alias stop="denv && dconda"
```


