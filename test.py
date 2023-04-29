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
from classes.Filters import Filters
import matplotlib.pyplot as plt
import numpy as np
from utils.plots import plot_eyetracking_filter, plot_participant_overview
from utils.normalisation import eye_tracking as normalise_eye_tracking
# Constants

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
