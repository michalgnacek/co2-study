# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:20:15 2023

@author: Michal Gnacek (www.gnacek.com)
"""

import neurokit2 as nk
import numpy as np

def calculate_hrv_features(df_ppg, sampling_frequency, return_plot=False):
    """
    Receives a dataframe with ppg data and returns
    time-domain features.
    """
    
    signals, info = nk.ppg_process(df_ppg, sampling_rate=sampling_frequency)
    peaks = signals.PPG_Peaks
    
    ## Plot summary
    # nk.ppg_plot(signals, sampling_rate=ORIG_SAMP_FREQUENCY_HZ)

    # Time-based features
    hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_frequency, show=False)

    # HRV features from neurokit2 that should be forwarded for final dataset
    HRV_SUBSET_FEATURES = ["HRV_MeanNN","HRV_SDNN","HRV_RMSSD","HRV_SDSD","HRV_MedianNN", "HRV_IQRNN"]
    hrv_time_features = hrv_time[ HRV_SUBSET_FEATURES ]
    
    # Calculate mean heart rate for the window
    hrv_time_features['Mean_BPM'] = np.mean(signals['PPG_Rate'])
    
    # Get values for actual peaks
    ppg_peaks = signals[signals['PPG_Peaks']==1]
    ppg_peaks = ppg_peaks.reset_index()
    
    for idx, peak in ppg_peaks.iterrows():
        #get number of rows between PPG peaks
        if(idx < (len(ppg_peaks)-2)):
            time_between_peaks = ppg_peaks.loc[idx+1]['index'] - ppg_peaks.loc[idx]['index']
            #with known frequency of 50hz, get seconds between peaks
            time_between_peaks = time_between_peaks * 0.02
            # estimate BPM with the time between peaks
            estimated_hr = 60/time_between_peaks
            # check if BPM estimation falls outside the 40 to 120 bpm range
            if ((estimated_hr < 40) or (estimated_hr > 120)):
                hrv_time_features[hrv_time_features.columns] = -1
                break

    # ## NOTE: Frequency features are not used because the window width may not enough for most features!
    # hrv_freq = nk.hrv_frequency(peaks, sampling_rate=ts_sampling_freq, show=True, normalize=True)
    # hrv_allfeatures = nk.hrv(peaks, sampling_rate=ts_sampling_freq, show=True)

    #### Save figure when `show=True`
    # save_path_plot = gen_path_plot(f"Preprocessing/PPG/Participant{participant}_{segment}")
    # fig = plt.gcf().set_size_inches(8, 5)
    # plt.tight_layout()
    # plt.savefig(save_path_plot)
    # plt.close()
    return hrv_time_features

def calculate_statistical_features(window, column_name):
    features = {}
    features[column_name + '_mean'] = np.mean(window)
    features[column_name + '_std'] = np.std(window)
    features[column_name + '_min'] = np.min(window)
    features[column_name + '_max'] = np.max(window)
    features[column_name + '_median'] = np.median(window)
    
    # Calculate maximum and minimum ratio
    max_value = np.max(window)
    min_value = np.min(window)
    features[column_name + '_max_ratio'] = max_value / np.abs(min_value)
    features[column_name + '_min_ratio'] = min_value / np.abs(max_value)
    
    # Calculate first and second derivatives
    first_derivative = np.gradient(window)
    second_derivative = np.gradient(first_derivative)
    features[column_name + '_1st_derivative_mean'] = np.mean(first_derivative)
    features[column_name + '_1st_derivative_std'] = np.std(first_derivative)
    features[column_name + '_2nd_derivative_mean'] = np.mean(second_derivative)
    features[column_name + '_2nd_derivative_std'] = np.std(second_derivative)
    
    return features
