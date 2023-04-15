# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 00:27:54 2023

@author: m
"""

import utils.constants as constants
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np

class Filters:
    
    
    
    def filter_fit_state_threshold(data):
        return data[data["Faceplate/FitState"] > constants.FIT_STATE_THRESHOLD]
    
    def filter_GSR(gsr_df):
        return gsr_df
    
    def filter_breathing(br_df):
        return br_df
    
    def filter_pupil_size(ps_df):
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