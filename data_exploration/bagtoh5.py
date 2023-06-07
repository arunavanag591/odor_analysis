import os
import pandas as pd
import multiprocessing as mp

import rosbag_pandas
from tqdm import tqdm 

pd.TimeSeries = pd.Series 

def clean_df(df):
    df.columns=df.columns.str.replace(r"/", "_")
    df = df[df.columns.drop(list(df.filter(regex='covariance')))]
    df = df[df.columns.drop(list(df.filter(regex='frame_id')))]
    df = df[df.columns.drop(list(df.filter(regex='twist_twist_angular')))]
    df = df[df.columns.drop(list(df.filter(regex='seq')))]
    return df

def rosbag_to_pandas(filename, topics):
    df = rosbag_pandas.bag_to_dataframe(filename, include=topics)
    df = clean_df(df)
    return df

def process_file(filename, folder_path, c):
    file_path = os.path.join(folder_path, filename)
    topics = ('/trisonica','/analog_output','/ublox_gps/fix', '/ublox_gps/fix_velocity','/imu/data') 
    df = rosbag_to_pandas(file_path, topics)
    df = df.reset_index()

    # df['_trisonica_header_frame_id'] = df['_trisonica_header_frame_id'].astype(str)
    df.to_hdf(folder_path +'h5/'+ str(filename.split('.')[0]) + '.h5', key='df', mode='w')



if __name__ == '__main__':
    folder_path = '../../data/BagsOctober/Run05/'
    file_list = os.listdir(folder_path)
   

    # Create a manager object to manage shared resources
    manager = mp.Manager()

    # Create a shared integer and set it to 0
    shared_counter = manager.Value('i', 0)

    # Set up a progress bar
    progress_bar = tqdm(total=len(file_list))

    # Create a pool of processes. By default, one is created for each CPU in your machine.
    with mp.Pool(mp.cpu_count()-4) as pool:
        for c, filename in enumerate(file_list):
            pool.apply_async(process_file, args=(filename, folder_path, c), callback=lambda _: progress_bar.update(1))
        pool.close()
        pool.join()

    progress_bar.close()
