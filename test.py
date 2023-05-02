# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:45:20 2023

@author: m
"""
# Imports
import os
import re
import csv
import pandas as pd
from utils.load_data import load_data_with_event_matching
from datetime import datetime
from classes.Participant import Participant
import matplotlib.pyplot as plt
import numpy as np
from utils.plots import plot_eyetracking_filter, plot_participant_overview, plot_assess_filter, plot_assess_filter2
from utils.normalisation import eye_tracking as normalise_eye_tracking
from classes.DataHandler import DataHandler
# Constants
import utils.constants as constants
from classes.Features import calculate_statistical_features
from utils.impute_eye_tracking_data import averge_fill
# Root directory for co2 data
CO2_DATA_DIRECTORY = r"D:\OneDrive - Bournemouth University\Studies\CO2 study\working_data"

#%%
from classes.DataHandler import DataHandler
air_mask_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\air\\2022-08-15T12-10-29.csv'
air_event_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\air\\2022-08-15T12-10-29.json'
air_eyetracking_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\air\\2022-08-15 121405.eyedata.csv'
air_biopac_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\air\\air.txt'
air_biopac_start_unix = 1660561855.0

air_mask_data = DataHandler.load_mask_data(air_mask_file, air_event_file, '10_benjamin')
air_eye_data = DataHandler.load_eyetracking_data(air_eyetracking_file, '10_benjamin', 'air')
air_biopac_data = DataHandler.load_biopac_data(air_biopac_file, air_biopac_start_unix, '10_benjamin')

co2_mask_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\co2\\2022-08-15T12-58-30.csv'
co2_event_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\co2\\2022-08-15T12-58-30.json'
co2_eyetracking_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\co2\\2022-08-15 130129.eyedata.csv'
co2_biopac_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\10_benjamin\\co2\\co2.txt'
co2_biopac_start_unix = 1660564699.0

co2_mask_data = DataHandler.load_mask_data(co2_mask_file, co2_event_file, '10_benjamin')
co2_eye_data = DataHandler.load_eyetracking_data(co2_eyetracking_file, '10_benjamin', 'co2')
co2_biopac_data = DataHandler.load_biopac_data(co2_biopac_file, co2_biopac_start_unix, '10_benjamin')


#%% LOAD SYNCED PARTICIAPNT DF
synced_participant_file = 'D:\\co2-study\\temp\\synced_participant_data\\63_reuben_moerman.csv'
participant_df = pd.read_csv(synced_participant_file)
#plot_participant_overview(participant_df, False)
#%%
participant_df = DataHandler.normalise_data(participant_df)
#%%
test = DataHandler.filter_data(participant_df)

#%% PUPIL SIZE
unfiltered_signal = participant_df[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]
filtered_signal = test[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]
#plot_assess_filter(unfiltered_signal[50000:50500], filtered_signal[50000:50500])
#plot_assess_filter(unfiltered_signal, filtered_signal)
#plot_assess_filter2(unfiltered_signal[50000:50500], filtered_signal[50000:50500])
plot_assess_filter2(unfiltered_signal, filtered_signal)

#%% GSR
unfiltered_signal = participant_df['Biopac_GSR']
filtered_signal = test['Biopac_GSR']
#plot_assess_filter(unfiltered_signal, filtered_signal)
#plot_assess_filter2(unfiltered_signal[50000:50500], filtered_signal[50000:50500])
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% RESPIRATION
unfiltered_signal = participant_df['Biopac_RSP']
filtered_signal = test['Biopac_RSP']
#plot_assess_filter(unfiltered_signal, filtered_signal)
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% PPG
unfiltered_signal = participant_df['Ppg/Raw.ppg']
filtered_signal = test['Ppg/Raw.ppg']
#plot_assess_filter(unfiltered_signal, filtered_signal)
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% EMG_CONTACT
unfiltered_signal = participant_df['Emg/Contact[CenterCorrugator]']
filtered_signal = test['Emg/Contact[CenterCorrugator]']
#plot_assess_filter(unfiltered_signal, filtered_signal)
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% ACC
unfiltered_signal = participant_df['Accelerometer/Raw.x']
filtered_signal = test['Accelerometer/Raw.x']
#plot_assess_filter(unfiltered_signal, filtered_signal)
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% GYR
unfiltered_signal = participant_df['Gyroscope/Raw.x']
filtered_signal = test['Gyroscope/Raw.x']
#plot_assess_filter(unfiltered_signal, filtered_signal)
plot_assess_filter2(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])



#%% feature extraction
features = DataHandler.extract_features(participant_df)
