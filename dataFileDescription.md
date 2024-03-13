# Data Files Walkthrough

## Abstract
Odor plumes in turbulent environments are intermittent and sparse. Lab-scaled experiments suggest that information about the source distance may be encoded in odor signal statistics, yet it is unclear whether useful and continuous distance estimates can be made under real-world flow conditions. Here we analyze odor signals from outdoor experiments with a sensor moving across large spatial scales in desert and forest environments to show that odor signal statistics can yield useful estimates of distance. We show that achieving accurate estimates of distance requires integrating statistics from 5-10 seconds, with a high temporal encoding of the olfactory signal of at least 20 Hz. By combining distance estimates from a linear model with wind-relative motion dynamics, we achieved source distance estimates in a 60x60 m$^2$ search area with median errors of 3-8 meters, a distance at which point odor sources are often within visual range for animals such as mosquitoes. 

## Data Description:

This contains all datasets involved in the analysis of the preprint as available here: ["Odor source location can be predicted from a time-history of odor statistics for a large-scale outdoor plume"](https://www.biorxiv.org/content/10.1101/2023.07.20.549973v1) and the analysis repository can be found in here in [github](https://github.com/arunavanag591/odor_analysis/tree/paper). All the file format are in pandas `.h5` or `.hdf` format, are compatible with `1.0.0 < pandas <= 1.5.3`


Download the data from Data dryad. The `data` folder along with `figure` folder, can be placed in the home folder under `~/odor_anaylsis/` .

The datasets can be divided into:
1. **Interpolated Sensor and stationery wind sensor data data **: These datasets include data from mobile sensor stack that contains imu, gps data and odor sensor and data from stationery wind sensors that has ambient wind velocity, direction. The interpolation is done with respect to the odor sensor's sampling speed. 

2. **Derived dataframes**: For different analysis example whiff statistics calculation, or lookback time analysis dataframes have been derived for easy to use plug and play experience.

### Folder Structure
```
├── data                       # Contains mobile sensor data from desert and forest
├── derived_data
   ├── KF                      # contains datasets for kalman filter analysis 
   ├── stationery_wind_data    # contains ambient wind sensor data from desert and forest
   ├── wind_lag_analysis       # contains data for wind lag analysis 
    
```

Below are the file descriptions under respective folders:

1. **data**
    - `ForestMASigned.h5` : Contains interpolated sensor data and signed distance axis for general Whittel Forest data analysis.
    - `NotWindyLR.h5` : Low resolution interpolated sensor data for wind speed `< 3m/s` for desert. 
    - `NotWindyMASigned.h5`: Contains high resolution interpolated clean sensor data for wind speed `< 3m/s` for desert. 
    - `WindyMASigned.h5`: Contains high resolution interpolated sensor data for wind speed `> 3m/s` for desert. 
2. **derived_data**
    - `NotWindyStatsTime_std.h5`: Contains derived lookback time whiff statistics for wind speed `< 3m/s` for desert.
    - `WindyStatsTime_std.h5`: Contains derived lookback time whiff statistics for wind speed `> 3m/s` for desert.
    - `ForestStatsTime_std.h5` : Contains derived lookback whiff statistics for forest.
    - `1hz.h5, 10hz.h5, 60hz.h5` : low pass filtered odor signal passed through 2nd order butterworth-filter and can be used with the script 8. [Figure 8](/data_exploration/figure/lowpassfilter.ipynb). 
    - `LpfForestFiltered.h5, LpfHWSFiltered.h5, LpfLWSFiltered.h5` :  datasets containing effect on R-squared value when odor signal is low pass filtered. Use with [Figure 8](/data_exploration/figure/lowpassfilter.ipynb). 
    - `HWSLTall.h5, LWSLTall.h5, ForestLTall.h5, R2LtTime.h5` :  datasets used in the script [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb)  to analyse the effect of lookback window on r-squared when correlated with distance.
    - `aic_filtered_model_params.h5` : Contains coefficients from the statsmodel of distance correlated aic filtered whiff statistics for all three wind scenarios. Run the script [Figure S9](/data_exploration/figure/Supplemental/windAicParamsAnalysis.ipynb). 
    - `lt_whiff_statistics.h5, All_AICDeltaTab.h5, all_Rsquared.h5, AllRsquaredAicCombinations.h5, `: Contains all the 25 whiff statistics calculated across the different distance from source over a look back time of 10 seconds. Run the script [Figure 5](/data_exploration/figure/figureAicR2layout.ipynb), in section `Bootstrapped R2` and `Boootstrapped R2 and AIC for filtered parameters` to see the results. 
   


## Data Fields
### Interpolated Sensor and stationery ambient sensor data
This section presents real-time and interpolated measurements from both mobile sensors on an agent and fixed ambient sensors, providing a comprehensive view of the environmental and agent-specific dynamics.

| Field Name                     | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `master_time`                  | Unix Time stamp from epoch                                                  |
| `time`                         | Time stamps starting from 0                                                 |
| `lon`                          | GPS longitude                                                               |
| `lat`                          | GPS latitude                                                                |
| `alt`                          | GPS altitude                                                                |
| `xsrc`                         | GPS coordinate converted to x axis in metres                                |
| `ysrc`                         | GPS coordinate converted to y axis in metres                                |
| `imu_angular_x`                | Angular velocity about the x-axis                                           |
| `imu_angular_y`                | Angular velocity about the y-axis                                           |
| `imu_angular_z`                | Angular velocity about the z-axis                                           |
| `imu_linear_acc_x`             | Linear acceleration along the x-axis                                        |
| `imu_linear_acc_y`             | Linear acceleration along the y-axis                                        |
| `imu_linear_acc_z`             | Linear acceleration along the z-axis                                        |
| `U`                            | East-west velocity of ambient wind (from stationary ground sensor)          |
| `V`                            | North-south velocity of ambient wind (from stationary ground sensor)        |
| `D`                            | X-Y ambient wind direction (from stationary ground sensor)                  |
| `S`                            | X-Y ambient wind magnitude (from stationary ground sensor)                  |
| `S2`                           | Speed of ambient wind magnitude (from stationary ground sensor)             |
| `corrected_u`                  | Ambient Wind velocity in east-west direction                                |
| `corrected_v`                  | Ambient Wind velocity in north-south direction                              |
| `nearest_from_streakline`      | Odor encounter in streakline coordinates in meters                          |
| `distance_along_streakline`    | Distance along streakline to odor encounter in meters                       |
| `distance_from_source`         | Distance from source to odor encounter in meters                            |
| `relative_parallel_comp`       | Relative parallel motion component of the agent with respect to wind        |
| `relative_perpendicular_comp`  | Relative parallel perpendicular component of the agent with respect to wind |

### Derived Analysis Data

This section details analytics derived from the raw and interpolated data, aimed at understanding patterns, frequencies, and statistical measures related to odor encounters and agent navigation relative to the source.

| Field Name                | Description                                                                                       |
|---------------------------|---------------------------------------------------------------------------------------------------|
| `type`                    | Classification of encounter distance from source (0: 0-5m, 1: 5-30m, 2: >30m)                     |
| `distance`                | Average distance over a look back time                                                            |
| `avg_dist_from_source`    | Averaged Distance from source in meters within a whiff                                            |
| `avg_dist_from_streakline`| Averaged Distance along streakline in meters within a whiff                                       |
| `mean_whiff_time`         | Average time duration of whiffs over a look back time                                             |
| `nwhiffs`                 | Number of whiff in a look back time                                                               |
| `efreq`                   | Encounter frequency in hz                                                                         |
| `mean_ef`                 | Average encounter frequency in hz                                                                 |
| `std_whiff`               | Mean standard deviation of odor within a whiff                                                    |
| `whiff_ma`                | Average moving average across a whiff                                                             |
| `mc_min`                  | Minimum of whiff concentration over a look back time                                              |
| `mc_max`                  | Maximum of whiff concentration over a look back time                                              |
| `mc_mean`                 | Avg of whiff concentration over a look back time                                                  |
| `mc_std_dev`              | Standard Deviation of Whiff concentration over a look back time                                   |
| `mc_k`                    | Kurtosis of whiff concentration over a look back time                                             |
| `wf_min`                  | Minimum of whiff frequency over a look back time                                                  |
| `wf_max`                  | Maximum of whiff frequency over a look back time                                                  |
| `wf_mean`                 | Avg of whiff frequency over a look back time                                                      |
| `wf_std_dev`              | Standard deviation of whiff frequency over a look back time                                       |
| `wf_k`                    | Kurtosis of whiff frequency over a look back time                                                 |
| `wd_min`                  | Minimum of whiff duration over a look back time                                                   |
| `wd_max`                  | Maximum of whiff duration over a look back time                                                   |
| `wd_mean`                 | Avg whiff duration over a look back time                                                          |
| `wd_std_dev`              | Standard deviation of Whiff Duration over a look back time                                        |
| `wd_k`                    | Kurtosis of Whiff Duration over a look back time                                                  |
| `ma_min`                  | Minimum of moving average over a look back time                                                   |
| `ma_max`                  | Maximum of moving average over a look back time                                                   |
| `ma_mean`                 | Average of moving average over a look back time                                                   |
| `ma_std_dev`              | Standard deviation of moving average over a look back time                                        |
| `ma_k`                    | Kurtosis of moving average over a look back time                                                  |
| `st_min`                  | Minimum of standard deviation over a look back time                                               |
| `st_max`                  | Maximum of standard deviation over a look back time                                               |
| `st_mean`                 | Average of standard deviation over a look back time                                               |
| `st_std_dev`              | Standard deviation of standard deviation over a look back time                                    |
| `st_k`                    | Kurtosis of standard deviation over a look back time                                              |


## Usage

This dataset is intended for use in developing and testing algorithms related to odor source localization in outdoor wind conditions in desert and forest terrains. Researchers are encouraged to utilize this data for exploring innovative approaches for odor source localization under varying environmental conditions.
