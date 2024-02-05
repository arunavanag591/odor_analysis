# Odor Analysis
This repository consist of the data analysis done for Odor Tracking experiment. 

Preprint: ["Odor source location can be predicted from a time-history of odor statistics for a large-scale outdoor plume"](https://www.biorxiv.org/content/10.1101/2023.07.20.549973v1)

## Dependencies

To visualize the below figures and see the results and calculations, you will need to install the following:
1. [FlyPlotLib](https://github.com/florisvb/FlyPlotLib)
2. [FigureFirst](https://github.com/FlyRanch/figurefirst)
3. [Inkscape 1.2](https://inkscape.org/release/inkscape-1.2/)

Follow the setup of [FigureFirst](https://github.com/FlyRanch/figurefirst) into inkscape.

other than these please check the section [Setup Section](#setup-environment) for the python libraries required

 - Please download the folders 'data' and 'figure' and place it inside the the folder [data_exploration](/data_exploration/), this should allow you to run all the scripts under the figures section.


## Figures

Data will be available in data dryad to run the results in the paper. Below are interactive notebooks, which can be used with Jupyter Notebook and run using python 3.8 and inskcape to generate the figures and results. These figures were generated using [figurefirst](https://github.com/FlyRanch/figurefirst).

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



## [Setup Environment](#setupheading)

Install <a href = "https://docs.python-guide.org/dev/virtualenvs/"> Virtualenv </a>: ```pip install virtualenv```<br/>

Install Anaconda from <a href = "https://docs.anaconda.com/anaconda/install/linux/">here. </a>



1. Create the virtualenv:

    ```
   virtualenv -p /usr/bin/python3.8 <env-name>  
   ```
  
2. Install Packages:

   ```
   pip install pandas
   pip install h5py
   pip install numpy
   pip install matplotlib
   pip install figurefirst
   pip install seaborn
   pip instal scikit-learn
   pip install h5py
   pip install tables
   python -m pip install statsmodels
   ``` 

