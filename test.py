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
from utils.plots import plot_eyetracking_filter
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
air_mask_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\2022-04-30T13-05-19.csv'
air_event_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\2022-04-30T13-05-19.json'
air_eyetracking_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\2022-04-30 131019.eyedata.csv'
air_biopac_file = 'D:\\OneDrive - Bournemouth University\\Studies\\CO2 study\\working_data\\2_john\\air\\air.txt' 
air_biopac_start_unix = 1651320511.0

air_mask_data = DataHandler.load_mask_data(air_mask_file, air_event_file, '2_john')
air_eye_data = DataHandler.load_eyetracking_data(air_eyetracking_file, '2_john', 'air')
air_biopac_data = DataHandler.load_biopac_data(air_biopac_file, air_biopac_start_unix, '2_john')

#%%
test = DataHandler.sync_signal_data(air_mask_data, air_eye_data, air_biopac_data, air_biopac_start_unix)

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
