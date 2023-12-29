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


#%%












import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, gc, gzip, pickle, json
from datetime import datetime

from tensorflow.keras.backend import clear_session
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from utils.STResNet import Classifier as STResNet
import tensorflow as tf


import warnings
warnings.filterwarnings('ignore')

#prinf libraries versions
print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)
print('tensorflow version: ', tf.__version__)
#python 3.9.15
import sklearn
print('sklearn version: ', sklearn.__version__)


#generate LSTM model with 2 lstm layers using keras functional API
def generate_LSTM (input_shape, output_shape, num_LSTM_layers=2, num_units=64, dropout=0.2, recurrent_dropout=0.2):
    clear_session()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    for i in range(num_LSTM_layers):
        x = tf.keras.layers.LSTM(num_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = tf.keras.layers.LSTM(num_units, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#generate ConvLSTM mode with 3 CNN layers and 1 LSTM layers using keras functional API
def generate_ConvLSTM (input_shape, output_shape, num_CNN_layers=3, num_units=64, dropout=0.2, recurrent_dropout=0.2):
    clear_session()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    for i in range(num_CNN_layers):
        x = tf.keras.layers.Conv1D(filters=num_units, kernel_size=3, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.LSTM(num_units, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = tf.keras.layers.Dense(num_units, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#read data
INFO_COLUMNS = ['Participant_No', 'Condition', 'Segment','Time']
EMG_AMP_COLUMNS = ['Emg/Amplitude[RightOrbicularis]',
               'Emg/Amplitude[RightZygomaticus]',
               'Emg/Amplitude[RightFrontalis]',
               'Emg/Amplitude[CenterCorrugator]',
               'Emg/Amplitude[LeftFrontalis]',
               'Emg/Amplitude[LeftZygomaticus]',
               'Emg/Amplitude[LeftOrbicularis]']
EMG_CONTACT_COLUMNS = ['Emg/Contact[RightOrbicularis]',
               'Emg/Contact[RightZygomaticus]',
               'Emg/Contact[RightFrontalis]',
               'Emg/Contact[CenterCorrugator]',
               'Emg/Contact[LeftFrontalis]',
               'Emg/Contact[LeftZygomaticus]',
               'Emg/Contact[LeftOrbicularis]']
HR_COLUMNS = ['HeartRate/Average', 'Ppg/Raw.ppg']
IMU_COLUMNS = ['Accelerometer/Raw.x', 'Accelerometer/Raw.y', 'Accelerometer/Raw.z', 
               'Gyroscope/Raw.x', 'Gyroscope/Raw.y', 'Gyroscope/Raw.z']
EYE_COLUMNS = ['VerboseData.Right.PupilDiameterMm','VerboseData.Left.PupilDiameterMm']
BIOPAC_RR_COLUMNS = ['Biopac_RSP']
BIOPAC_GSR_COLUMNS = ['Biopac_GSR']
SENSOR_COLUMNS = EMG_AMP_COLUMNS + EMG_CONTACT_COLUMNS + HR_COLUMNS + IMU_COLUMNS + EYE_COLUMNS + BIOPAC_RR_COLUMNS + BIOPAC_GSR_COLUMNS

ALL_COLUMNS = INFO_COLUMNS + SENSOR_COLUMNS
SAMPLING_RATE = 50 #original sampling is 50, bu later we downsample to 10
data_folder = 'temp/synced_participant_data/'

#%%

df_emg_amp = pd.DataFrame()
df_emg_contact = pd.DataFrame()
df_hr = pd.DataFrame()
df_imu = pd.DataFrame()
df_eye = pd.DataFrame()
df_rr = pd.DataFrame()
df_gsr = pd.DataFrame()
df_all_sensor = pd.DataFrame() #all senosr data
df_info = pd.DataFrame() #all info data including labels

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        df = pd.read_csv(data_folder + file)
        #original sampling is 50, downsample to 10
        df = df.iloc[::5, :]
        df = df[ALL_COLUMNS]
        #dropna
        print('before dropna: ', df.shape)
        df = df.dropna()
        print('after dropna: ', df.shape)

        df_all_sensor = pd.concat([df_all_sensor, df[SENSOR_COLUMNS]])
        df_info = pd.concat([df_info, df[INFO_COLUMNS]])

        df_emg_amp = pd.concat([df_emg_amp, df[EMG_AMP_COLUMNS]])
        df_emg_contact = pd.concat([df_emg_contact, df[EMG_CONTACT_COLUMNS]])
        df_hr = pd.concat([df_hr, df[HR_COLUMNS]])
        df_imu = pd.concat([df_imu, df[IMU_COLUMNS]])
        df_eye = pd.concat([df_eye, df[EYE_COLUMNS]])
        df_rr = pd.concat([df_rr, df[BIOPAC_RR_COLUMNS]])
        df_gsr = pd.concat([df_gsr, df[BIOPAC_GSR_COLUMNS]])

        
        del df
        
#%%

df_all_sensor = df_all_sensor.reset_index(drop=True)
df_info = df_info.reset_index(drop=True)


df_emg_amp = df_emg_amp.reset_index(drop=True)
df_emg_contact = df_emg_contact.reset_index(drop=True)
df_hr = df_hr.reset_index(drop=True)
df_imu = df_imu.reset_index(drop=True)
df_eye = df_eye.reset_index(drop=True)
df_rr = df_rr.reset_index(drop=True)
df_gsr = df_gsr.reset_index(drop=True)



df_all_sensor.shape, df_info.shape, df_emg_amp.shape, df_emg_contact.shape, df_hr.shape, df_imu.shape, df_eye.shape, df_rr.shape, df_gsr.shape

#%%

print(df_all_sensor.isnull().sum())
df_all_sensor.describe().round(2)

#%%

df_info.columns

#%%

df_info.Condition.value_counts().plot(kind='bar')
plt.title('Number of samples per condition')
plt.show()

plt.figure(figsize=(20, 5))
df_info.Participant_No.value_counts().plot(kind='bar')
plt.title('Number of samples per participant')
plt.show()

#%%

#sliding window segmentation
WINDOW_SIZE_SECONDS = 30
WINDOW_SIZE = WINDOW_SIZE_SECONDS * SAMPLING_RATE
STEP_SIZE = (WINDOW_SIZE_SECONDS//6) * SAMPLING_RATE #get prediction every 5 seconds
#perform segmentation for each user and each condition
segments_all_sensor_dict = {}
segments_emg_amp_dict = {}
segments_emg_contact_dict = {}
segments_hr_dict = {}
segments_imu_dict = {}
segments_eye_dict = {}
segments_rr_dict = {}
segments_gsr_dict = {}
segments_labels_dict = {}

for user in df_info.Participant_No.unique():
    user_segments_all_sensor = []
    user_segments_emg_amp_sensor = []
    user_segments_emg_contact_sensor = []
    user_segments_hr_sensor = []
    user_segments_imu_sensor = []
    user_segments_eye_sensor = []
    user_segments_rr_sensor = []
    user_segments_gsr_sensor = []
    user_labels = []
    for condition in df_info.Condition.unique():
        user_cond_all_sensor_df = df_all_sensor[(df_info.Participant_No == user) & (df_info.Condition == condition)]
        user_cond_emg_amp_df = user_cond_all_sensor_df[EMG_AMP_COLUMNS]
        user_cond_emg_contact_df = user_cond_all_sensor_df[EMG_CONTACT_COLUMNS]
        user_cond_hr_df = user_cond_all_sensor_df[HR_COLUMNS]
        user_cond_imu_df = user_cond_all_sensor_df[IMU_COLUMNS]
        user_cond_eye_df = user_cond_all_sensor_df[EYE_COLUMNS]
        user_cond_rr_df = user_cond_all_sensor_df[BIOPAC_RR_COLUMNS]
        user_cond_gsr_df = user_cond_all_sensor_df[BIOPAC_GSR_COLUMNS]
        user_cond_info_df = df_info[(df_info.Participant_No == user) & (df_info.Condition == condition)]
        
        for i in range(0, len(user_cond_all_sensor_df) - WINDOW_SIZE, STEP_SIZE):
            user_segments_all_sensor.append(user_cond_all_sensor_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_emg_amp_sensor.append(user_cond_emg_amp_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_emg_contact_sensor.append(user_cond_emg_contact_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_hr_sensor.append(user_cond_hr_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_imu_sensor.append(user_cond_imu_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_eye_sensor.append(user_cond_eye_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_rr_sensor.append(user_cond_rr_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_segments_gsr_sensor.append(user_cond_gsr_df.iloc[i:i+WINDOW_SIZE, :].values)
            user_labels.append(user_cond_info_df.Condition.iloc[i:i+WINDOW_SIZE].values)
            
    segments_all_sensor_dict[user] = user_segments_all_sensor
    
    segments_emg_amp_dict[user] = user_segments_emg_amp_sensor
    segments_emg_contact_dict[user] = user_segments_emg_contact_sensor
    segments_hr_dict[user] = user_segments_hr_sensor
    segments_imu_dict[user] = user_segments_imu_sensor
    segments_eye_dict[user] = user_segments_eye_sensor
    segments_rr_dict[user] = user_segments_rr_sensor
    segments_gsr_dict[user] = user_segments_gsr_sensor
    
    segments_labels_dict[user] = user_labels
    #print segments and labels shape
    print('user {} segments shape: '.format(user), np.array(user_segments_all_sensor).shape)
    print('user {} labels shape: '.format(user), np.array(user_labels).shape)
    print()

del df_all_sensor, df_info


































