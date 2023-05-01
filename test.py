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
from utils.plots import plot_eyetracking_filter, plot_participant_overview, plot_assess_filter
from utils.normalisation import eye_tracking as normalise_eye_tracking
from classes.DataHandler import DataHandler
# Constants
import utils.constants as constants
from classes.Features import calculate_statistical_features

# Root directory for co2 data
CO2_DATA_DIRECTORY = r"D:\OneDrive - Bournemouth University\Studies\CO2 study\working_data"

#%%
from classes.DataHandler import DataHandler
eye_df = DataHandler.load_eyetracking_data('D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\2022-04-30 131019.eyedata.csv', '2_john', 'air')

#%%
from classes.DataHandler import DataHandler
eye_df = DataHandler.load_eyetracking_data('D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\2022-04-30 131019.eyedata.csv', '2')
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

#%%
from classes.DataHandler import DataHandler
air_synced_signal = DataHandler.sync_signal_data(air_mask_data, air_eye_data, air_biopac_data, air_biopac_start_unix)
#%%
co2_synced_signal = DataHandler.sync_signal_data(co2_mask_data, co2_eye_data, co2_biopac_data, co2_biopac_start_unix)
#%%
test = DataHandler.label_data(co2_synced_signal.copy())

#%%
co2_mask_data = DataHandler.load_mask_data(co2_mask_file, co2_event_file, '3_karolina')

#%%

downsampled_data = DataHandler.downsample_participant_data('2_john', DataHandler.label_data(air_synced_signal), DataHandler.label_data(co2_synced_signal))
#%%
air_eye_data = DataHandler.load_eyetracking_data(air_eyetracking_file, '2_john', 'air')

#%%
import matplotlib.pyplot as plt
#plt.plot(co2_synced_signal['VerboseData.Left.PupilDiameterMm'])
#plt.plot(air_synced_signal['VerboseData.Left.PupilDiameterMm'])
#plt.plot(air_synced_signal['Biopac_GSR'])
#plt.plot(co2_synced_signal['Biopac_GSR'])
plt.plot(downsampled_data['Biopac_GSR'])
#%%
bla = test.copy()
bla = DataHandler.label_data(bla)
#%%
filtered_data = Filters.filter_pupil_size(air_eye_data)
plot_eyetracking_filter(air_eye_data['VerboseData.Left.PupilDiameterMm'], filtered_data['VerboseData.Left.PupilDiameterMm'])
#%%
filtered_data = Filters.filter_pupil_size(test)
plot_eyetracking_filter(test['VerboseData.Left.PupilDiameterMm'], filtered_data['VerboseData.Left.PupilDiameterMm'])
#%%

import numpy as np
import matplotlib.pyplot as plt

# Example signal data
signal = air_eye_data['VerboseData.Left.PupilDiameterMm']

# Define the window size for the moving average filter
window_size = 3600
window_size = 7200

# Create the moving average filter kernel
kernel = np.ones(window_size) / window_size

# Apply the moving average filter to the signal. Mode same ensures length of output is same as input, meaning window slide of 1 data row
filtered_signal = np.convolve(signal, kernel, mode='same')

# Plot the original and filtered signals
plt.plot(signal, label='Original signal')
plt.plot(filtered_signal, label='Filtered signal')
plt.legend()
plt.show()

#%%
from classes.DataHandler import DataHandler
df = pd.read_csv(r'D:\co2-study\temp\synced_participant_data\4_raff.csv')
normalised_data = DataHandler.normalise_data(df)
#%%
import matplotlib.pyplot as plt

plot_participant_overview(normalised_data, True)
#%%
from utils.timestamps import read_unix, j2000_to_unix

#%%
plot_participant_overview(normalised_data)
#%%
first_time_value = normalised_data.loc[normalised_data['Segment'] == 'gas_inhalation', 'Time'].iloc[0]
last_time_value = normalised_data.loc[normalised_data['Segment'] == 'gas_inhalation', 'Time'].iloc[-1]
time = last_time_value - first_time_value
print(time/60)

#%%
read_unix(1652106477.43704)

#%%
synced_participant_file = 'D:\\co2-study\\temp\\synced_participant_data\\63_reuben_moerman.csv'
participant_df = pd.read_csv(synced_participant_file)
participant_df = DataHandler.normalise_data(participant_df)
#%%
test = DataHandler.filter_data(participant_df)
#test['Segment'] = test['Segment'].fillna('setup')

#%% PUPIL SIZE
import matplotlib.pyplot as plt

unfiltered_signal = participant_df[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]
filtered_signal = test[constants.DATA_COLUMNS.EYE_LEFT_PUPIL_SIZE.value]
plot_assess_filter(unfiltered_signal, filtered_signal)

#%% GSR
import matplotlib.pyplot as plt

unfiltered_signal = participant_df['Biopac_GSR']
filtered_signal = test['Biopac_GSR']
plot_assess_filter(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% RESPIRATION
import matplotlib.pyplot as plt

unfiltered_signal = participant_df['Biopac_RSP']
filtered_signal = test['Biopac_RSP']
plot_assess_filter(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])

#%% EMG_CONTACT
import matplotlib.pyplot as plt

unfiltered_signal = participant_df['Emg/Contact[CenterCorrugator]']
filtered_signal = test['Emg/Contact[CenterCorrugator]']
plot_assess_filter(unfiltered_signal, filtered_signal)
#plt.plot(test['Biopac_GSR'][test['Condition']=='CO2'])
#plt.plot(test['Biopac_GSR'][(test['Condition'] == 'CO2') & (test['Segment'] == 'brightness_calibration')])



#%% feature extraction
features = DataHandler.extract_features(participant_df)

#%%
import pandas as pd
import numpy as np
from utils.generate_sliding_windows import generate_sliding_windows

# Assume you have a DataFrame called 'df' with the columns mentioned in your question
df = test.copy()

columns_to_calculate = ['Biopac_GSR', 'Biopac_RSP']  # Specify the columns to calculate features for

windows = generate_sliding_windows(df, 5, 3)
result = pd.DataFrame()
for window in windows:
    window_features = pd.DataFrame()
    #for column in window[columns_to_calculate]:
    for column_name, column_data in window[columns_to_calculate].iteritems(): 
        features = calculate_statistical_features(column_data, column_name)
        if(window_features.empty):
            window_features = pd.DataFrame([features])
        else:
            window_features = pd.concat([window_features, pd.DataFrame([features])], axis=1)
        #result = result.append(features, ignore_index=True)
    result = pd.concat([result, window_features], ignore_index=True)



#%% feature extraction

import pandas as pd
import numpy as np
from utils.generate_sliding_windows import generate_sliding_windows

# Assume you have a DataFrame called 'df' with the columns mentioned in your question
df = test.copy()

columns_to_calculate = ['Biopac_GSR', 'Biopac_RSP']  # Specify the columns to calculate features for

statistical_features = []

for column in columns_to_calculate:
    windows = generate_sliding_windows(df[column], 5, 3)  # Generate sliding windows for the current column

    for window in windows:
        features = {}
        features[column + '_Mean'] = window.mean()
        features[column + '_Standard Deviation'] = window.std()
        features[column + '_Maximum'] = window.max()
        # Add more statistical features as needed
        
        statistical_features.append(features)

# Create a dataframe from the list of dictionaries
result_df = pd.DataFrame(statistical_features)


#%% feature extraction
features_test = test.copy()
import pandas as pd
import numpy as np

# Assume you have a DataFrame called 'df' with the columns mentioned in your question
df = test.copy()
window_length = 5  # Length of the sliding window in seconds
shift = 3  # Shift between consecutive windows in seconds

# Convert shift and window_length to the corresponding number of rows based on the sampling frequency of the data
sampling_frequency = 50  # Replace with your actual sampling frequency
shift_rows = int(shift * sampling_frequency)
window_length_rows = int(window_length * sampling_frequency)

# Generate sliding windows using rolling window functionality for each column except 'Participant_No', 'Condition', 'Segment'
columns_to_process = [col for col in df.columns if col not in ['Participant_No', 'Condition', 'Segment', 'Frame#', 'Time',
       'Faceplate/FaceState', 'Faceplate/FitState', 'unix_timestamp', 'Event']]
windows = df[columns_to_process].rolling(window=window_length_rows, min_periods=window_length_rows, center=False)

# Extract statistical features from each window
features = windows.agg(['mean', 'std', 'min', 'max'])  # Add additional statistical functions as needed

# Reset index to represent the start time of each window
features.reset_index(drop=True, inplace=True)

# Create a DataFrame with the 'Participant_No', 'Condition', 'Segment', and 'Frame#' columns for reference
reference_columns = df[['Participant_No', 'Condition', 'Segment', 'Frame#']]

# Concatenate the reference columns with the statistical features
output_df = pd.concat([reference_columns, features], axis=1)

# Print the resulting DataFrame with statistical features
print(output_df)

