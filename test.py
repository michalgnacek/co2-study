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
from utils.plots import Plots
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
#Plots.participant_overview(participant_df, False)

#%% feature extraction

features = DataHandler.extract_features(participant_df)

#%% feature extraction

features = DataHandler.extract_features(pd.read_csv('E:\\co2-study\\temp\\synced_participant_data\\2_john.csv'))

#%%

import neurokit2 as nk

# Download data
data = nk.data("bio_resting_8min_100hz")

# Process the data
df, info = nk.eda_process(data["EDA"], sampling_rate=100)

# Single dataframe is passed
nk.eda_intervalrelated(df)

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
synced_participant_file = pd.read_csv('D:\\co2-study\\temp\\synced_participant_data\\23_peter_h.csv')

#%%
test = DataHandler.extract_features_entire_condition(synced_participant_file)
#test = DataHandler.extract_features(synced_participant_file)

#%%
# File containing features for entire segments
segment_features_file = 'D:\\co2-study\\temp\\segment_features.csv'
segment_features = pd.read_csv(segment_features_file, index_col=0)

# File containing features for windows of data
windowed_features_file = 'D:\\co2-study\\temp\\windowed_features.csv'
windowed_features = pd.read_csv(windowed_features_file, index_col=0)

#%%
test = pd.read_csv('D:\\co2-study\\temp\\gas_inhalation_df.csv')
test2 = pd.read_csv('D:\\co2-study\\temp\\gas_inhalation_df2.csv')


