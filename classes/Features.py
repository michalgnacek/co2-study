# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:20:15 2023

@author: Michal Gnacek (www.gnacek.com)
"""

import neurokit2 as nk
import numpy as np

def calculate_hrv_features(df_ppg, sampling_frequency):
    """
    Receives a dataframe with ppg data and returns
    time-domain features.
    """
    exclude_hr_range = False
    
    signals, info = nk.ppg_process(df_ppg, sampling_rate=sampling_frequency)
    
    ppg_features = nk.ppg_analyze(signals, sampling_rate=50)

    # HRV features from neurokit2 that should be forwarded for final dataset
    SUBSET_FEATURES = ["PPG_Rate_Mean","HRV_MeanNN","HRV_SDNN","HRV_RMSSD","HRV_SDSD","HRV_MedianNN", "HRV_IQRNN"]
    ppg_features = ppg_features[ SUBSET_FEATURES ]
        
    if(exclude_hr_range):
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
                    ppg_features[ppg_features.columns] = -1
                    break

    return ppg_features

def calculate_rsp_features(df_rsp, sampling_frequency):
    """
    Receives a dataframe with biopac rsp data and returns
    time-domain features.
    """
    if(df_rsp.isna().any()):
        return np.nan
    else:
        signals, info = nk.rsp_process(df_rsp, sampling_rate=sampling_frequency)
        
        rsp_features = nk.rsp_analyze(signals, sampling_rate=50)

        # HRV features from neurokit2 that should be forwarded for final dataset
        SUBSET_FEATURES = ['RSP_Rate_Mean', 'RSP_Amplitude_Mean', 'RSP_Phase_Duration_Ratio']
        rsp_features = rsp_features[ SUBSET_FEATURES ]
        
        return rsp_features

def calculate_gsr_features(df_gsr, sampling_frequency):
    """
    Receives a dataframe with biopac gsr data and returns
    time-domain features.
    """
    if(df_gsr.isna().any()):
        return np.nan
    else:
        signals, info = nk.eda_process(df_gsr, sampling_rate=sampling_frequency)
        
        gsr_features = nk.eda_analyze(signals, sampling_rate=50)

        # HRV features from neurokit2 that should be forwarded for final dataset
        #SUBSET_FEATURES = ['RSP_Rate_Mean', 'RSP_Amplitude_Mean', 'RSP_Phase_Duration_Ratio']
        #gsr_features = gsr_features[ SUBSET_FEATURES ]
        
        return gsr_features

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
    
    if(column_name=='Ppg/Raw.ppg'):
        hrv_time_features = calculate_hrv_features(window, 50)
        for hrv_feature_name, hrv_feature_data in hrv_time_features.iteritems(): 
            features[hrv_feature_name] = hrv_feature_data[0]
    
    if(column_name=='Biopac_RSP'):
        rsp_time_features = calculate_rsp_features(window, 50)
        for rsp_feature_name, rsp_feature_data in rsp_time_features.iteritems(): 
            features[rsp_feature_name] = rsp_feature_data[0]
            
    if(column_name=='Biopac_GSR'):
        gsr_time_features = calculate_gsr_features(window, 50)
        for gsr_feature_name, gsr_feature_data in gsr_time_features.iteritems(): 
            features[gsr_feature_name] = gsr_feature_data[0]
    
    return features
