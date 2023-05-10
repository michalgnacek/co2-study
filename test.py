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
air_mask_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\air\\2022-09-04T19-10-25.csv'
air_event_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\air\\2022-09-04T19-10-25.json'
air_eyetracking_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\air\\2022-09-04 191406.eyedata.csv'
air_biopac_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\air\\air.txt'
air_biopac_start_unix = 1662315007.0

air_mask_data = DataHandler.load_mask_data(air_mask_file, air_event_file, '28_atul_singh')
air_eye_data = DataHandler.load_eyetracking_data(air_eyetracking_file, '28_atul_singh', 'air')
air_biopac_data = DataHandler.load_biopac_data(air_biopac_file, air_biopac_start_unix, '28_atul_singh')

co2_mask_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\co2\\2022-09-04T19-57-08.csv'
co2_event_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\co2\\2022-09-04T19-57-08.json'
co2_eyetracking_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\co2\\2022-09-04 200014.eyedata.csv'
co2_biopac_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\28_atul_singh\\co2\\co2.txt'
co2_biopac_start_unix = 1662317813.0

co2_mask_data = DataHandler.load_mask_data(co2_mask_file, co2_event_file, '28_atul_singh')
co2_eye_data = DataHandler.load_eyetracking_data(co2_eyetracking_file, '28_atul_singh', 'co2')
co2_biopac_data = DataHandler.load_biopac_data(co2_biopac_file, co2_biopac_start_unix, '28_atul_singh')

#%%
co2_eye_data = DataHandler.load_eyetracking_data(co2_eyetracking_file, '28_atul_singh', 'co2')
#air_synced_data = DataHandler.sync_signal_data(air_mask_data, air_eye_data, air_biopac_data, air_biopac_start_unix))

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

#%%
synced_participant_file = 'D:\\co2-study\\temp\\synced_participant_data\\52_thomas_charnock.csv'
#features_file = 'D:\\co2-study\\temp\\features\\7_aliaksei.csv'
participant_df = pd.read_csv(synced_participant_file)
#features = pd.read_csv(features_file,index_col=False)
#condition_features = features[features['Segment']=='gas_inhalation']
#condition_features.to_csv('D:\\co2-study\\temp\\test.csv')

#%%
DataHandler.merge_participant_windowed_feature_files()

#%%
DataHandler.merge_participant_data_files()

#%%
DataHandler.merge_participant_segment_feature_files()

#%%
features_file = 'D:\\co2-study\\temp\\segment_features.csv'
features = pd.read_csv(features_file, index_col=0)
#%%
signal_file = 'D:\\co2-study\\temp\\combined_data.csv'
signals = pd.read_csv(signal_file)

#%%
synced_participant_file = pd.read_csv('D:\\co2-study\\temp\\synced_participant_data\\2_john.csv')

#%%
test = DataHandler.extract_features_entire_condition(synced_participant_file)
#test = DataHandler.extract_features(synced_participant_file)

#%%
features_file = 'D:\\co2-study\\temp\\segment_features.csv'
features = pd.read_csv(features_file, index_col=0)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with the desired columns

# Filter the data for 'air' and 'co2' conditions with 'gas_inhalation' segment
filtered_data = features[(features['Condition'].isin(['AIR', 'CO2'])) & (features['Segment'] == 'gas_inhalation')]

# Create the violin plot
sns.violinplot(x='Condition', y='Biopac_GSR_mean', data=filtered_data)

# Customize plot titles and labels
sns.set(style='whitegrid')
plt.title('Mean GSR for Air and CO2 Conditions')
plt.xlabel('Condition')
plt.ylabel('GSR Mean')

# Display the plot
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with the desired columns

# Filter the data for 'air' and 'co2' conditions with 'gas_inhalation' segment
filtered_data = features[(features['Condition'].isin(['AIR', 'CO2'])) & (features['Segment'] == 'gas_inhalation')]

# Create the violin plot
sns.violinplot(x='Condition', y='RSP_Rate_Mean', data=filtered_data)

# Customize plot titles and labels
sns.set(style='whitegrid')
plt.title('Mean Respiratory Rate for Air and CO2 Conditions')
plt.xlabel('Condition')
plt.ylabel('Mean Respiration')

# Display the plot
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with the desired columns

# Filter the data for 'air' and 'co2' conditions with 'gas_inhalation' segment
filtered_data = features[(features['Condition'].isin(['AIR', 'CO2'])) & (features['Segment'] == 'gas_inhalation')]

# Create the violin plot
sns.violinplot(x='Condition', y='VerboseData.Left.PupilDiameterMm_mean', data=filtered_data)

# Customize plot titles and labels
sns.set(style='whitegrid')
plt.title('Mean Pupil Size for Air and CO2 Conditions')
plt.xlabel('Condition')
plt.ylabel('Mean Pupil Size')

# Display the plot
plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with the desired columns

# Filter the data for 'air' and 'co2' conditions with 'gas_inhalation' segment
filtered_data = features[(features['Condition'].isin(['AIR', 'CO2'])) & (features['Segment'] == 'gas_inhalation')]

# Create the violin plot
sns.violinplot(x='Condition', y='PPG_Rate_Mean', data=filtered_data)

# Customize plot titles and labels
sns.set(style='whitegrid')
plt.title('Mean Heart Rate for Air and CO2 Conditions')
plt.xlabel('Condition')
plt.ylabel('Mean Heart Rate')

# Display the plot
plt.show()

#%%
# File containing features for entire segments
segment_features_file = 'D:\\co2-study\\temp\\segment_features.csv'
segment_features = pd.read_csv(segment_features_file, index_col=0)

# File containing features for windows of data
windowed_features_file = 'D:\\co2-study\\temp\\windowed_features.csv'
windowed_features = pd.read_csv(windowed_features_file, index_col=0)
#%%
filtered_data = features[(features['Condition'].isin(['AIR', 'CO2'])) & (features['Segment'] == 'gas_inhalation')]


#%%

from utils.plots import plot_features_time_series, contact_gsr_line_plot

# File containing features for entire segments
segment_features_file = 'D:\\co2-study\\temp\\segment_features.csv'
segment_features = pd.read_csv(segment_features_file, index_col=0)

# File containing features for windows of data
windowed_features_file = 'D:\\co2-study\\temp\\windowed_features.csv'
windowed_features = pd.read_csv(windowed_features_file, index_col=0)

# Filter the data for 'air' and 'co2' conditions with 'gas_inhalation' segment
gas_inhalation_segments = segment_features[(segment_features['Condition'].isin(['AIR', 'CO2'])) & (segment_features['Segment'] == 'gas_inhalation')]
gas_inhalation_windows = windowed_features[(windowed_features['Condition'].isin(['AIR', 'CO2'])) & (windowed_features['Segment'] == 'gas_inhalation')]

gas_inhalation_segments['pupil_size_combined'] = (gas_inhalation_segments['VerboseData.Left.PupilDiameterMm_mean'] + gas_inhalation_segments['VerboseData.Right.PupilDiameterMm_mean']) / 2
gas_inhalation_windows['pupil_size_combined'] = (gas_inhalation_windows['VerboseData.Left.PupilDiameterMm_mean'] + gas_inhalation_windows['VerboseData.Right.PupilDiameterMm_mean']) / 2


# add window index for each participant/condition
window_index = pd.DataFrame({'window_index': gas_inhalation_windows.groupby(['participant_number', 'Condition']).cumcount()})
gas_inhalation_windows.insert(3, 'window_index', window_index['window_index'])
#%%



muscles = ['LeftOrbicularis','RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']

for muscle in muscles:
    muscle_name = f'Emg/Contact[{muscle}]_mean'
    plot_title = f'{muscle} Mean Over Time'
    contact_gsr_line_plot(gas_inhalation_windows, muscle_name, plot_title)
