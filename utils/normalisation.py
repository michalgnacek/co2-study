# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:49:58 2023

@author: m
"""
from sklearn.preprocessing import MinMaxScaler
import utils.constants as constants
import pandas as pd

def eye_tracking(participant_df):
    normalised_eye_data = participant_df.copy()
    #TODO: Normalise Left pupul size
    left_eye_min_max_scaler = MinMaxScaler()
    left_eye_min_max_scaler.fit(participant_df[[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]])
    normalised_eye_data[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value] = left_eye_min_max_scaler.transform(normalised_eye_data[[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]])
    
    #TODO: Normalise Right pupul size
    right_eye_min_max_scaler = MinMaxScaler()
    right_eye_min_max_scaler.fit(participant_df[[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value]])
    normalised_eye_data[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value] = right_eye_min_max_scaler.transform(normalised_eye_data[[constants.DATA_COLUMNS.EYE_RIGHT_PUPIL_SIZE.value]])
    return normalised_eye_data

def min_max_normalisation(data_to_normalise):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(pd.DataFrame(data_to_normalise))
    return min_max_scaler.transform(pd.DataFrame(data_to_normalise))
