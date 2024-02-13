# Data Files Walkthrough

## Abstract
Odor plumes in turbulent environments are intermittent and sparse. Lab-scaled experiments suggest that information about the source distance may be encoded in odor signal statistics, yet it is unclear whether useful and continuous distance estimates can be made under real-world flow conditions. Here we analyze odor signals from outdoor experiments with a sensor moving across large spatial scales in desert and forest environments to show that odor signal statistics can yield useful estimates of distance. We show that achieving accurate estimates of distance requires integrating statistics from 5-10 seconds, with a high temporal encoding of the olfactory signal of at least 20 Hz. By combining distance estimates from a linear model with wind-relative motion dynamics, we achieved source distance estimates in a 60x60 m$^2$ search area with median errors of 3-8 meters, a distance at which point odor sources are often within visual range for animals such as mosquitoes. 

## Data Description:

This contains all datasets involved in the analysis of the preprint as available here: ["Odor source location can be predicted from a time-history of odor statistics for a large-scale outdoor plume"](https://www.biorxiv.org/content/10.1101/2023.07.20.549973v1) and the analysis repository can be found in here in [github](https://github.com/arunavanag591/odor_analysis/tree/paper). All the file format are in pandas `.h5` or `.hdf` format, are compatible with `1.0.0 < pandas <= 1.5.3`


Download the data from Data dryad. The `data` folder along with `figure` folder, can be placed in the home folder under `~/odor_anaylsis/` .

The datasets can be divided into:
1. **Interpolated Sensor data**: These datasets include data from mobile sensor stack that contained imu, gps data and odor sensor. The interpolation is done with respect to the odor sensor's sampling speed. 

2. **Stationery Ambient wind sensor data**: These data comes from the stationery wind sensors that provides ambient wind velocity and direction. 

3. **Derived dataframes**: For different analysis example whiff statistics calculation, or lookback time analysis dataframes have been derived for easy to use plug and play experience.

Below are the file and folder descriptions:

- `aic_filtered_model_params.h5` : Contains coefficients from the statsmodel of distance correlated aic filtered whiff statistics for all three wind scenarios. Run the script [Figure S9](/data_exploration/figure/Supplemental/windAicParamsAnalysis.ipynb). 
- `DesertWind` : Folder containing data from the stationery wind sensor measuring ambient wind.
- `ForestMASigned.h5` : Contains interpolated clean sensor data and signed distance axis for general Whittel Forest data analysis.
- `ForestStatsTime_std.h5` : Contains derived lookback whiff statistics for forest.
- `KF`: Contains all the datasets required for the kalman filter analysis as in script [Figure 7](/data_exploration/figure/klmfigure.ipynb) and [Figure S10](/data_exploration/figure/Supplemental/klmsupplemental.ipynb). 
- `LookbackTimeAnalysis`: Folder contains dataset used in the script [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb)  to analyse the effect of lookback window on r-squared when correlated with distance.
- `lowpassfilter`: contains dataframes containing odor signal passed throw 2nd order low-pass-filter and can be used with the script 8. [Figure 8](/data_exploration/figure/lowpassfilter.ipynb). 
- `lt_whiff_statistics.h5`: Contains all the 25 whiff statistics calculated across the different distance from source over a look back time of 10 seconds. Run the script [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb), in section `Bootstrapped R2` and `Boootstrapped R2 and AIC for filtered parameters` to see the results. 
- `methodfigure1`: Contains all the datasets and images required to reproduce the experimental setup in Black Rock Desert, in the script [Figure 1](/data_exploration/figure/method1.ipynb).
- `methodfigure2`: Contains all the datasets and images required to reproduce the experimental setup in Whittel Forest, in the script [Figure 2](/data_exploration/figure/method2.ipynb).
1. `NotWindy.h5` : Contains low resolution interpolated clean sensor data for wind speed `< 3m/s` for desert. 
1. `NotWindyMASigned.h5`: Contains high resolution interpolated clean sensor data for wind speed `< 3m/s` for desert. 
1. `NotWindyStatsTime_std.h5`: Contains derived lookback time whiff statistics for wind speed `< 3m/s` for desert.
1. `R2_AIC`: Folder contains r-squared and delta aic value when bootstrapped. Run the script [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb) to see the results.
1. `windLagAnalysis` : Folder containing all the files required to analysis for wind 
1. `WindyMASigned.h5`: Contains high resolution interpolated sensor data for wind speed `> 3m/s` for desert. 
1. `WindyStatsTime_std.h5`: Contains derived lookback time whiff statistics for wind speed `> 3m/s` for desert.
