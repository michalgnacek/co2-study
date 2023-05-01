# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:13:51 2023

@author: m
"""

def generate_sliding_windows(data, window_length, shift):
    
    sampling_frequency = 50  # Replace with your actual sampling frequency
    shift = int(shift * sampling_frequency)
    window_length = int(window_length * sampling_frequency)
    
    windows = []
    num_rows = data.shape[0]
    start = 0
    end = window_length
    
    while end <= num_rows:
        window = data[start:end]
        windows.append(window)
        start += shift
        end += shift
    
    return windows