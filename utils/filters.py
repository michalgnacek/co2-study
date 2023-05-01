# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 00:27:54 2023

@author: m
"""

import utils.constants as constants
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np


    
    

def filter_fit_state_threshold(data):
    return data[data["Faceplate/FitState"] > constants.FIT_STATE_THRESHOLD]
    
def filter_pupil_size2(ps_df):
    #TODO: update to filter both pupils
    data = ps_df[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]
    
    """
    Moving average filter on 30 secs (3600 frames at 120hz) time windows
    """
    
    #window_size = 3600
    #1min wind size at 120hz
    #window_size = 7200
    #1min wind size at 1000hz
    window_size = 60000
    

    # Create the moving average filter kernel
    kernel = np.ones(window_size) / window_size

    # Apply the moving average filter to the signal
    filtered_data = np.convolve(data, kernel, mode='same')

    """
    Apply a Butterworth band-pass filter to data (disabled)
    """
    # Define filter parameters
    nyq = 0.5 * constants.FREQUENCIES.EYE_TRACKING.value
    low = 0.02 / nyq
    high = 4 / nyq
    order = 3
    
    # Create the filter coefficients
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter
    #filtered_data = filtfilt(b, a, filtered_data)

    filtered_data = pd.DataFrame(filtered_data, columns = [constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value])
    
    return filtered_data

def butter_lowpass_filter(signal, cutoff, fs, order):
    # Normalize the cutoff frequency
    #normalized_cutoff = cutoff / (0.5 * fs)

    # Design the Butterworth filter
    b, a = butter(order, cutoff, btype='low', analog=True, output='ba')

    # Apply zero-phase forward-backward filtering
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def filter_pupil_size(participant_df):
    cut_off = 0.1
    fs = 50
    order = 3
    
    left_pupil_air_data = participant_df[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value][participant_df['Condition']=='AIR']
    left_pupil_air_data = left_pupil_air_data[left_pupil_air_data.notna()]
    filtered_array_air = butter_lowpass_filter(left_pupil_air_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_left_pupil_data_air = pd.DataFrame(data=filtered_array_air, columns=[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value], index=left_pupil_air_data.index)
    # Update values in 'participant_df' using filtered 'left_pupil_data'
    participant_df.update(filtered_left_pupil_data_air)
    
    left_pupil_co2_data = participant_df[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value][participant_df['Condition']=='CO2']
    left_pupil_co2_data = left_pupil_co2_data[left_pupil_co2_data.notna()]
    filtered_array_co2 = butter_lowpass_filter(left_pupil_co2_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_left_pupil_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value], index=left_pupil_co2_data.index)
    # Update values in 'participant_df' using filtered 'left_pupil_data'
    participant_df.update(filtered_left_pupil_data_co2)
    
    right_pupil_air_data = participant_df[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value][participant_df['Condition']=='AIR']
    right_pupil_air_data = right_pupil_air_data[right_pupil_air_data.notna()]
    filtered_array_air = butter_lowpass_filter(right_pupil_air_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_right_pupil_data_air = pd.DataFrame(data=filtered_array_air, columns=[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value], index=right_pupil_air_data.index)
    # Update values in 'participant_df' using filtered 'right_pupil_data'
    participant_df.update(filtered_right_pupil_data_air)
    
    right_pupil_co2_data = participant_df[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value][participant_df['Condition']=='CO2']
    right_pupil_co2_data = right_pupil_co2_data[right_pupil_co2_data.notna()]
    filtered_array_co2 = butter_lowpass_filter(right_pupil_co2_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_right_pupil_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value], index=right_pupil_co2_data.index)
    # Update values in 'participant_df' using filtered 'right_pupil_data'
    participant_df.update(filtered_right_pupil_data_co2)

    return participant_df

def filter_biopac_gsr(participant_df):
    cut_off = 0.3
    fs = 50
    order = 3
    
    gsr_air_data = participant_df['Biopac_GSR'][participant_df['Condition']=='AIR']
    gsr_air_data = gsr_air_data[gsr_air_data.notna()]
    filtered_array_air = butter_lowpass_filter(gsr_air_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_gsr_data_air = pd.DataFrame(data=filtered_array_air, columns=['Biopac_GSR'], index=gsr_air_data.index)
    # Update values in 'participant_df' using filtered 'left_pupil_data'
    participant_df.update(filtered_gsr_data_air)
    
    gsr_co2_data = participant_df['Biopac_GSR'][participant_df['Condition']=='CO2']
    gsr_co2_data = gsr_co2_data[gsr_co2_data.notna()]
    filtered_array_co2 = butter_lowpass_filter(gsr_co2_data, cut_off, fs, order)
    # Create a DataFrame from the filtered array
    filtered_gsr_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=['Biopac_GSR'], index=gsr_co2_data.index)
    # Update values in 'participant_df' using filtered 'left_pupil_data'
    participant_df.update(filtered_gsr_data_co2)
    
    return participant_df

def filter_biopac_respiration(participant_df):    
    cut_off = 0.3
    fs = 50
    order = 3
    
    respiration_air_data = participant_df['Biopac_RSP'][participant_df['Condition']=='AIR']
    respiration_air_data = respiration_air_data[respiration_air_data.notna()]
    filtered_array_air = butter_lowpass_filter(respiration_air_data, cut_off, fs, order)
    filtered_respiration_data_air = pd.DataFrame(data=filtered_array_air, columns=['Biopac_RSP'], index=respiration_air_data.index)
    participant_df.update(filtered_respiration_data_air)
    
    respiration_co2_data = participant_df['Biopac_RSP'][participant_df['Condition']=='CO2']
    respiration_co2_data = respiration_co2_data[respiration_co2_data.notna()]
    filtered_array_co2 = butter_lowpass_filter(respiration_co2_data, cut_off, fs, order)
    filtered_respiration_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=['Biopac_RSP'], index=respiration_co2_data.index)
    participant_df.update(filtered_respiration_data_co2)
    
    return participant_df

def filter_ppg(participant_df):    
    cut_off = 0.3
    fs = 50
    order = 3
    
    ppg_air_data = participant_df['Ppg/Raw.ppg'][participant_df['Condition']=='AIR']
    ppg_air_data = ppg_air_data[ppg_air_data.notna()]
    filtered_array_air = butter_lowpass_filter(ppg_air_data, cut_off, fs, order)
    filtered_ppg_data_air = pd.DataFrame(data=filtered_array_air, columns=['Ppg/Raw.ppg'], index=ppg_air_data.index)
    participant_df.update(filtered_ppg_data_air)
    
    ppg_co2_data = participant_df['Ppg/Raw.ppg'][participant_df['Condition']=='CO2']
    ppg_co2_data = ppg_co2_data[ppg_co2_data.notna()]
    filtered_array_co2 = butter_lowpass_filter(ppg_co2_data, cut_off, fs, order)
    filtered_ppg_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=['Ppg/Raw.ppg'], index=ppg_co2_data.index)
    participant_df.update(filtered_ppg_data_co2)
    
    return participant_df

def filter_contact(participant_df):    
    cut_off = 0.3
    fs = 50
    order = 3
    
    for emg_contact_column in participant_df[constants.DATA_COLUMNS.EMG_CONTACT.value]:
        emg_contact_data_air = participant_df[emg_contact_column][participant_df['Condition']=='AIR']
        emg_contact_data_air = emg_contact_data_air[emg_contact_data_air.notna()]
        filtered_array_air = butter_lowpass_filter(emg_contact_data_air, cut_off, fs, order)
        filtered_emg_contact_data_air = pd.DataFrame(data=filtered_array_air, columns=[emg_contact_column], index=emg_contact_data_air.index)
        participant_df.update(filtered_emg_contact_data_air)
        
        emg_contact_data_co2 = participant_df[emg_contact_column][participant_df['Condition']=='CO2']
        emg_contact_data_co2 = emg_contact_data_co2[emg_contact_data_co2.notna()]
        filtered_array_co2 = butter_lowpass_filter(emg_contact_data_co2, cut_off, fs, order)
        filtered_emg_contact_data_co2 = pd.DataFrame(data=filtered_array_co2, columns=[emg_contact_column], index=emg_contact_data_co2.index)
        participant_df.update(filtered_emg_contact_data_co2)
        
    return participant_df

    

    
    