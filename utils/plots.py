# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 02:23:12 2023

@author: m
"""

import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import math

class Plots:
    
    def __init__(self):
    # Constructor or initialization method
        pass

    def eyetracking_filter(raw_signal, filtered_signal, participant_id, condition):
        # Plot the original and filtered signals
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(raw_signal, 'b-', label='Raw Signal')
        ax[0].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Pupil size (mm)')
        ax[0].legend()
    
        ax[1].plot(filtered_signal, 'g-', linewidth=2, label='Filtered Signal')
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Pupil size (mm)')
        ax[1].legend()
        
        plt.title('Pupil size for ' + condition + ' condition. Participant: ' + participant_id)
        plt.tight_layout()
        plt.show()
        
    def participant_overview(participant_df, save_plot):
    
        PLOT_COLUMNS = ['HeartRate/Average', 'Ppg/Raw.ppg', 
                        'VerboseData.Right.PupilDiameterMm', 'VerboseData.Left.PupilDiameterMm', 
                        'Biopac_GSR', 'Biopac_RSP']
        
        GSR_air = participant_df['Biopac_GSR'][(participant_df['Condition'] == 'AIR')]
        GSR_co2 = participant_df['Biopac_GSR'][(participant_df['Condition'] == 'CO2')]
        
        RSP_air = participant_df['Biopac_RSP'][(participant_df['Condition'] == 'AIR')]
        RSP_co2 = participant_df['Biopac_RSP'][(participant_df['Condition'] == 'CO2')]
        
        pupil_size_left_air = participant_df['VerboseData.Left.PupilDiameterMm'][(participant_df['Condition'] == 'AIR')]
        pupil_size_right_air = participant_df['VerboseData.Right.PupilDiameterMm'][(participant_df['Condition'] == 'AIR')]
        pupil_size_left_co2 = participant_df['VerboseData.Left.PupilDiameterMm'][(participant_df['Condition'] == 'CO2')]
        pupil_size_right_co2 = participant_df['VerboseData.Right.PupilDiameterMm'][(participant_df['Condition'] == 'CO2')]
      
        HR_AVG_air = participant_df['HeartRate/Average'][(participant_df['Condition'] == 'AIR')]
        HR_AVG_co2 = participant_df['HeartRate/Average'][(participant_df['Condition'] == 'CO2')]
        
        PPG_RAW_air = participant_df['Ppg/Raw.ppg'][(participant_df['Condition'] == 'AIR')]
        PPG_RAW_co2 = participant_df['Ppg/Raw.ppg'][(participant_df['Condition'] == 'CO2')]
        
        EMG_C_RO_AIR = participant_df['Emg/Contact[RightOrbicularis]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_RZ_AIR = participant_df['Emg/Contact[RightZygomaticus]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_RF_AIR = participant_df['Emg/Contact[RightFrontalis]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_CC_AIR = participant_df['Emg/Contact[CenterCorrugator]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_LF_AIR = participant_df['Emg/Contact[LeftFrontalis]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_LZ_AIR = participant_df['Emg/Contact[LeftZygomaticus]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_LO_AIR = participant_df['Emg/Contact[LeftOrbicularis]'][(participant_df['Condition'] == 'AIR')]
        EMG_C_RO_CO2 = participant_df['Emg/Contact[RightOrbicularis]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_RZ_CO2 = participant_df['Emg/Contact[RightZygomaticus]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_RF_CO2 = participant_df['Emg/Contact[RightFrontalis]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_CC_CO2 = participant_df['Emg/Contact[CenterCorrugator]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_LF_CO2 = participant_df['Emg/Contact[LeftFrontalis]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_LZ_CO2 = participant_df['Emg/Contact[LeftZygomaticus]'][(participant_df['Condition'] == 'CO2')]
        EMG_C_LO_CO2 = participant_df['Emg/Contact[LeftOrbicularis]'][(participant_df['Condition'] == 'CO2')]
        
        EMG_A_RO_AIR = participant_df['Emg/Amplitude[RightOrbicularis]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_RZ_AIR = participant_df['Emg/Amplitude[RightZygomaticus]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_RF_AIR = participant_df['Emg/Amplitude[RightFrontalis]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_CC_AIR = participant_df['Emg/Amplitude[CenterCorrugator]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_LF_AIR = participant_df['Emg/Amplitude[LeftFrontalis]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_LZ_AIR = participant_df['Emg/Amplitude[LeftZygomaticus]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_LO_AIR = participant_df['Emg/Amplitude[LeftOrbicularis]'][(participant_df['Condition'] == 'AIR')]
        EMG_A_RO_CO2 = participant_df['Emg/Amplitude[RightOrbicularis]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_RZ_CO2 = participant_df['Emg/Amplitude[RightZygomaticus]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_RF_CO2 = participant_df['Emg/Amplitude[RightFrontalis]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_CC_CO2 = participant_df['Emg/Amplitude[CenterCorrugator]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_LF_CO2 = participant_df['Emg/Amplitude[LeftFrontalis]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_LZ_CO2 = participant_df['Emg/Amplitude[LeftZygomaticus]'][(participant_df['Condition'] == 'CO2')]
        EMG_A_LO_CO2 = participant_df['Emg/Amplitude[LeftOrbicularis]'][(participant_df['Condition'] == 'CO2')]
        
        ACC_X_air = participant_df['Accelerometer/Raw.x'][(participant_df['Condition'] == 'AIR')]
        ACC_Y_air = participant_df['Accelerometer/Raw.y'][(participant_df['Condition'] == 'AIR')]
        ACC_Z_air = participant_df['Accelerometer/Raw.z'][(participant_df['Condition'] == 'AIR')]
        
        ACC_X_co2 = participant_df['Accelerometer/Raw.x'][(participant_df['Condition'] == 'CO2')]
        ACC_Y_co2 = participant_df['Accelerometer/Raw.y'][(participant_df['Condition'] == 'CO2')]
        ACC_Z_co2 = participant_df['Accelerometer/Raw.z'][(participant_df['Condition'] == 'CO2')]
        
        GYR_X_air = participant_df['Gyroscope/Raw.x'][(participant_df['Condition'] == 'AIR')]
        GYR_Y_air = participant_df['Gyroscope/Raw.y'][(participant_df['Condition'] == 'AIR')]
        GYR_Z_air = participant_df['Gyroscope/Raw.z'][(participant_df['Condition'] == 'AIR')]
        
        GYR_X_co2 = participant_df['Gyroscope/Raw.x'][(participant_df['Condition'] == 'CO2')]
        GYR_Y_co2 = participant_df['Gyroscope/Raw.y'][(participant_df['Condition'] == 'CO2')]
        GYR_Z_co2 = participant_df['Gyroscope/Raw.z'][(participant_df['Condition'] == 'CO2')]
       
        
        
        # Assume GSR_air, GSR_co2, BR_air, BR_co2 are NumPy arrays containing the signal values
        # Assume condition_1 and condition_2 are strings containing the names of the conditions
        
        # Set up the figure with 2 rows and 2 columns of subplots
        fig, axs = plt.subplots(9, 2, figsize=(10, 8))
        
        # Plot GSR
        axs[0, 0].plot(GSR_air)
        axs[0, 0].set_title("GSR - AIR")
        
        axs[0, 1].plot(GSR_co2)
        axs[0, 1].set_title("GSR - CO2")
        
        # Plot Respiration
        axs[1, 0].plot(RSP_air)
        axs[1, 0].set_title("RSP - AIR")
        
        axs[1, 1].plot(RSP_co2)
        axs[1, 1].set_title("RSP - CO2")
        
        # Plot Pupil size
        axs[2, 0].plot(pupil_size_left_air, label='left')
        axs[2, 0].plot(pupil_size_right_air, label='right')
        axs[2, 0].set_title("Pupil Size - AIR")
        axs[2, 0].legend()
        
        axs[2, 1].plot(pupil_size_left_co2, label='left')
        axs[2, 1].plot(pupil_size_right_co2, label='right')
        axs[2, 1].set_title("Pupil Size - CO2")
        axs[2, 1].legend()
        
        # HR Average
        axs[3, 0].plot(HR_AVG_air)
        axs[3, 0].set_title("HR/AVG - AIR")
        
        axs[3, 1].plot(HR_AVG_co2)
        axs[3, 1].set_title("HR/AVG - CO2")
        
        # PPG Raw
        axs[4, 0].plot(PPG_RAW_air)
        axs[4, 0].set_title("PPG Raw - AIR")
        
        axs[4, 1].plot(PPG_RAW_co2)
        axs[4, 1].set_title("PPG Raw - CO2")
        
        # PPG Raw
        axs[4, 0].plot(PPG_RAW_air)
        axs[4, 0].set_title("PPG Raw - AIR")
    
        axs[4, 1].plot(PPG_RAW_co2)
        axs[4, 1].set_title("PPG Raw - CO2")
    
        # EMG Contact
        axs[5, 0].plot(EMG_C_RO_AIR, label='Emg/Contact[RightOrbicularis]')
        axs[5, 0].plot(EMG_C_RZ_AIR, label='Emg/Contact[RightZygomaticus]')
        axs[5, 0].plot(EMG_C_RF_AIR, label='Emg/Contact[RightFrontalis]')
        axs[5, 0].plot(EMG_C_CC_AIR, label='Emg/Contact[CenterCorrugator]')
        axs[5, 0].plot(EMG_C_LF_AIR, label='Emg/Contact[LeftFrontalis]')
        axs[5, 0].plot(EMG_C_LZ_AIR, label='Emg/Contact[LeftZygomaticus]')
        axs[5, 0].plot(EMG_C_LO_AIR, label='Emg/Contact[LeftOrbicularis]')
        axs[5, 0].set_title("EMG Contact - AIR")
        
        axs[5, 1].plot(EMG_C_RO_CO2, label='Emg/Contact[RightOrbicularis]')
        axs[5, 1].plot(EMG_C_RZ_CO2, label='Emg/Contact[RightZygomaticus]')
        axs[5, 1].plot(EMG_C_RF_CO2, label='Emg/Contact[RightFrontalis]')
        axs[5, 1].plot(EMG_C_CC_CO2, label='Emg/Contact[CenterCorrugator]')
        axs[5, 1].plot(EMG_C_LF_CO2, label='Emg/Contact[LeftFrontalis]')
        axs[5, 1].plot(EMG_C_LZ_CO2, label='Emg/Contact[LeftZygomaticus]')
        axs[5, 1].plot(EMG_C_LO_CO2, label='Emg/Contact[LeftOrbicularis]')
        axs[5, 1].set_title("EMG Contact - CO2")
    
        # EMG AMP
        axs[6, 0].plot(EMG_A_RO_AIR, label='Emg/Amplitude[RightOrbicularis]')
        axs[6, 0].plot(EMG_A_RZ_AIR, label='Emg/Amplitude[RightZygomaticus]')
        axs[6, 0].plot(EMG_A_RF_AIR, label='Emg/Amplitude[RightFrontalis]')
        axs[6, 0].plot(EMG_A_CC_AIR, label='Emg/Amplitude[CenterCorrugator]')
        axs[6, 0].plot(EMG_A_LF_AIR, label='Emg/Amplitude[LeftFrontalis]')
        axs[6, 0].plot(EMG_A_LZ_AIR, label='Emg/Amplitude[LeftZygomaticus]')
        axs[6, 0].plot(EMG_A_LO_AIR, label='Emg/Amplitude[LeftOrbicularis]')
        axs[6, 0].set_title("EMG Amplitude - AIR")
    
        axs[6, 1].plot(EMG_A_RO_CO2, label='Emg/Amplitude[RightOrbicularis]')
        axs[6, 1].plot(EMG_A_RZ_CO2, label='Emg/Amplitude[RightZygomaticus]')
        axs[6, 1].plot(EMG_A_RF_CO2, label='Emg/Amplitude[RightFrontalis]')
        axs[6, 1].plot(EMG_A_CC_CO2, label='Emg/Amplitude[CenterCorrugator]')
        axs[6, 1].plot(EMG_A_LF_CO2, label='Emg/Amplitude[LeftFrontalis]')
        axs[6, 1].plot(EMG_A_LZ_CO2, label='Emg/Amplitude[LeftZygomaticus]')
        axs[6, 1].plot(EMG_A_LO_CO2, label='Emg/Amplitude[LeftOrbicularis]')
        axs[6, 1].set_title("EMG Amplitude - CO2")
        
        # Plot ACC
        axs[7, 0].plot(ACC_X_air, label='X')
        axs[7, 0].plot(ACC_Y_air, label='Y')
        axs[7, 0].plot(ACC_Z_air, label='Z')
        axs[7, 0].set_title("Acc - AIR")
        axs[7, 0].legend()
        
        axs[7, 1].plot(ACC_X_co2, label='X')
        axs[7, 1].plot(ACC_Y_co2, label='Y')
        axs[7, 1].plot(ACC_Z_co2, label='Z')
        axs[7, 1].set_title("Acc - CO2")
        axs[7, 1].legend()
        
        # Plot GYR
        axs[8, 0].plot(GYR_X_air, label='X')
        axs[8, 0].plot(GYR_Y_air, label='Y')
        axs[8, 0].plot(GYR_Z_air, label='Z')
        axs[8, 0].set_title("Gyr - AIR")
        axs[8, 0].legend()
        
        axs[8, 1].plot(GYR_X_co2, label='X')
        axs[8, 1].plot(GYR_Y_co2, label='Y')
        axs[8, 1].plot(GYR_Z_co2, label='Z')
        axs[8, 1].set_title("Gyr - CO2")
        axs[8, 1].legend()
    
        fig.tight_layout()
        plt.suptitle('Participant: ' + participant_df.loc[0]['Participant_No'])
        if save_plot is True:
            plot_par_directory = os.path.join(os.getcwd(), 'temp', 'plots')
            plot_child_directory = os.path.join(os.getcwd(), 'temp', 'plots', 'normalised_signal_overview')
            if not os.path.exists(plot_par_directory):
                os.mkdir(plot_par_directory)
                if not os.path.exists(plot_child_directory):
                    os.mkdir(plot_child_directory)
            plt.savefig(os.path.join(plot_child_directory, participant_df.loc[0]['Participant_No']) + '.png', dpi=300)
        plt.show()
        
    def assess_filter(unfiltered_signal, filtered_signal):
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
        # Plot the unfiltered signal
        ax1.plot(unfiltered_signal, label='Unfiltered')
        ax1.set_ylabel('Signal')
        ax1.set_xlabel('Time')
        ax1.set_title('Unfiltered')
    
        # Plot the filtered signal
        ax2.plot(filtered_signal, label='Filtered')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Signal')
        ax2.set_title('Filtered')
    
        # Link the x-axis limits of the two subplots
        ax1.get_shared_x_axes().join(ax1, ax2)
    
    
        # Show the plot
        plt.show()
        
    def assess_filter2(unfiltered_signal, filtered_signal):
        
        plt.figure(figsize=(10, 6))
        plt.plot(unfiltered_signal, label='Original Signal')
        plt.plot(filtered_signal, label='Filtered Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Butterworth Low-pass Filter')
        plt.legend()
        plt.show()
    
    def features_time_series(windowed_features, feature_column, title, xlabel, ylabel, plot_path=False, 
                                  air_prediction_line=[], co2_prediction_line=[]):
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
        #mean_gsr['condition_index'] = mean_gsr.groupby('Condition').cumcount()
        
        # Scale the x-axis values to a range between 0 and 20
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
        
        
        # Compute the standard error of the mean
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition']=='AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition']=='CO2']
        
        # Set the plot style
        sns.set(style='whitegrid')
        
        # Plot the line plot
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
            
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_air[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_air[feature_column], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_co2[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_co2[feature_column], alpha=0.2)
        
        # Customize plot titles and labels
        plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(np.arange(0, 21, 5))
        plt.xlim([0, 20])
        
        if(not len(co2_prediction_line)==0):
            plt.plot((co2_prediction_line[0] * 20 / 114), co2_prediction_line[1], color='red', label='CO2 Fitted', linestyle=(0, (5, 10)))
            
        if(not len(air_prediction_line)==0):
            plt.plot((air_prediction_line[0] * 20 / 114), air_prediction_line[1], color='blue', label='Air Fitted', linestyle=(0, (5, 10)))
        
        #if(not plot_path):
          #   plt.savefig(plot_path)
        plt.savefig(plot_path)
        
        # Display the plot
        plt.show()
        
        # Define the logistic function
                
    def features_time_series_gsr(windowed_features, feature_column, title, xlabel, ylabel, air_function, co2_function, plot_path=False, 
                                  ):
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
        
        # Scale the x-axis values to a range between 0 and 20
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
        
        
        # Compute the standard error of the mean
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition']=='AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition']=='CO2']
        
        # Set the plot style
        sns.set(style='whitegrid')
        
        # Plot the line plot
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
            
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_air[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_air[feature_column], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_co2[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_co2[feature_column], alpha=0.2)
        
        
        # CO2 Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_co2 = windowed_features['window_index'][windowed_features['Condition'] == 'CO2']
        y_co2 = windowed_features[feature_column][windowed_features['Condition'] == 'CO2']
        
        
        # Adjusted bounds and initial guess
        bounds_co2 = ([0, 0, 0], [np.max(y_co2), np.max(x_co2), np.max(x_co2)])
        initial_guess_co2 = [0.660, 0.161, 3.126]
        
        popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, bounds=bounds_co2, p0=initial_guess_co2)
        
        
        #popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, bounds=([0, 0, 0], [np.max(y_co2), np.max(x_co2), np.max(x_co2)]))
        
        a, b, c = popt_co2
        # Print the logistic equation
        print(f"CO2 Logistic Equation: y = {a} / (1 + e^(-{b} * (x - {c})))")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_co2 = np.linspace(min(x_co2), max(x_co2), 1000)  # Adjust the number of points as needed
        fit_line_co2 = co2_function(x_fit_co2, *popt_co2)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_co2 = (x_fit_co2 - min(x_fit_co2)) / (max(x_fit_co2) - min(x_fit_co2)) * 20
        
        # Plot only the logistic curve fit
        plt.plot(x_fit_scaled_co2, fit_line_co2, '--', color='red', label='Logistic Fit', linewidth='3')
        
        
        # AIR Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_air = windowed_features['window_index'][windowed_features['Condition'] == 'AIR']
        y_air = windowed_features[feature_column][windowed_features['Condition'] == 'AIR']
        popt_air, _ = curve_fit(air_function, x_air, y_air, bounds=([0, 0, 0], [np.max(y_air), np.max(x_air), np.max(x_air)]))
        
        a, b, c = popt_air
        # Print the logistic equation
        print(f"AIR Quadratic Equation: y = {a} * (x - {c})^2 + {b}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_air = np.linspace(min(x_air), max(x_air), 1000)  # Adjust the number of points as needed
        fit_line_air = air_function(x_fit_air, *popt_air)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_air = (x_fit_air - min(x_fit_air)) / (max(x_fit_air) - min(x_fit_air)) * 20
        
        # Plot only the logistic curve fit
        plt.plot(x_fit_scaled_air, fit_line_air, '--', color='blue', label='Quadratic Fit', linewidth='3')
        
        
        
        # Customize plot titles and labels
        #plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(np.arange(0, 21, 5))
        plt.xlim([0, 20])
    
        
        #if(not plot_path):
          #   plt.savefig(plot_path)
        plt.savefig(plot_path)
        
        # Display the plot
        plt.show()
        
    def features_time_series_rr(windowed_features, feature_column, title, xlabel, ylabel, air_function, co2_function, plot_path=False):
          
        
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
        
        # Scale the x-axis values to a range between 0 and 20
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
        
        # Compute the standard error of the mean
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition']=='AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition']=='CO2']
        
        # Set the plot style
        sns.set(style='whitegrid')
        
        # Plot the line plot
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
            
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_air[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_air[feature_column], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_co2[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_co2[feature_column], alpha=0.2)
        
        
        # CO2 Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_co2 = windowed_features['window_index'][windowed_features['Condition'] == 'CO2']
        y_co2 = windowed_features[feature_column][windowed_features['Condition'] == 'CO2']
        initial_guess = [np.max(y_co2), 0.1, np.median(x_co2)]
        popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, p0=initial_guess)
        #popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, bounds=([0, 0, 0], [np.max(y_co2), np.max(x_co2), np.max(x_co2)]))
        
        a, b, c = popt_co2
        # Print the logistic equation
        print(f"CO2 Logistic Equation: y = {a} / (1 + e^(-{b} * (x - {c})))")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_co2 = np.linspace(min(x_co2), max(x_co2), 1000)  # Adjust the number of points as needed
        fit_line_co2 = co2_function(x_fit_co2, *popt_co2)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_co2 = (x_fit_co2 - min(x_fit_co2)) / (max(x_fit_co2) - min(x_fit_co2)) * 20
        
        # Plot only the logistic curve fit
        plt.plot(x_fit_scaled_co2, fit_line_co2, '--', color='red', label='Logistic Fit', linewidth='3')
        
        
        # AIR Condition FIT
        # Fit the straight line function to the data with adjusted bounds
        x_air = windowed_features['window_index'][windowed_features['Condition'] == 'AIR']
        y_air = windowed_features[feature_column][windowed_features['Condition'] == 'AIR']
        popt_air, _ = curve_fit(air_function, x_air, y_air, bounds=([0, 0], [np.max(y_air), np.max(x_air)]))
        
        m_air, c_air = popt_air
        # Print the straight line equation
        print(f"AIR Straight Line Equation: y = {m_air} * x + {c_air}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_air = np.linspace(min(x_air), max(x_air), 1000)  # Adjust the number of points as needed
        fit_line_air = air_function(x_fit_air, *popt_air)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_air = (x_fit_air - min(x_fit_air)) / (max(x_fit_air) - min(x_fit_air)) * 20
        
        # Plot only the straight line fit
        plt.plot(x_fit_scaled_air, fit_line_air, '--', color='blue', label='Straight Line Fit', linewidth=3)
        
        
        
        # Customize plot titles and labels
        #plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(np.arange(0, 21, 5))
        plt.xlim([0, 20])
        
        # Save or display the plot
        if plot_path:
            plt.savefig(plot_path)
        plt.show()
        
    def features_time_series_hr(windowed_features, feature_column, title, xlabel, ylabel, air_function, co2_function, plot_path=False, 
                                  ):
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
        
        # Scale the x-axis values to a range between 0 and 20
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
        
        
        # Compute the standard error of the mean
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition']=='AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition']=='CO2']
        
        # Set the plot style
        sns.set(style='whitegrid')
        
        # Plot the line plot
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
            
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_air[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_air[feature_column], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_co2[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_co2[feature_column], alpha=0.2)
        
        
        # CO2 Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_co2 = windowed_features['window_index'][windowed_features['Condition'] == 'CO2']
        y_co2 = windowed_features[feature_column][windowed_features['Condition'] == 'CO2']
        popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, bounds=([0, 0], [np.max(y_co2), np.max(x_co2)]))
        
        m_co2, c_co2 = popt_co2
        # Print the straight line equation
        print(f"CO2 Straight Line Equation: y = {m_co2} * x + {c_co2}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_co2 = np.linspace(min(x_co2), max(x_co2), 1000)  # Adjust the number of points as needed
        fit_line_co2 = co2_function(x_fit_co2, *popt_co2)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_co2 = (x_fit_co2 - min(x_fit_co2)) / (max(x_fit_co2) - min(x_fit_co2)) * 20
        
        # Plot only the straight line fit
        plt.plot(x_fit_scaled_co2, fit_line_co2, '--', color='red', label='Straight Line Fit', linewidth=3)
        
        
        # AIR Condition FIT
        # Fit the straight line function to the data with adjusted bounds
        x_air = windowed_features['window_index'][windowed_features['Condition'] == 'AIR']
        y_air = windowed_features[feature_column][windowed_features['Condition'] == 'AIR']
        popt_air, _ = curve_fit(air_function, x_air, y_air, bounds=([0, 0], [np.max(y_air), np.max(x_air)]))
        
        m_air, c_air = popt_air
        # Print the straight line equation
        print(f"AIR Straight Line Equation: y = {m_air} * x + {c_air}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_air = np.linspace(min(x_air), max(x_air), 1000)  # Adjust the number of points as needed
        fit_line_air = air_function(x_fit_air, *popt_air)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_air = (x_fit_air - min(x_fit_air)) / (max(x_fit_air) - min(x_fit_air)) * 20
        
        # Plot only the straight line fit
        plt.plot(x_fit_scaled_air, fit_line_air, '--', color='blue', label='Straight Line Fit', linewidth=3)
        
        
        
        # Customize plot titles and labels
        #plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(np.arange(0, 21, 5))
        plt.xlim([0, 20])
    
        
        #if(not plot_path):
          #   plt.savefig(plot_path)
        plt.savefig(plot_path)
        
        # Display the plot
        plt.show()
        
    def features_time_series_pupilsize(windowed_features, feature_column, title, xlabel, ylabel, air_function, co2_function, plot_path=False, 
                                  ):
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
        
        # Scale the x-axis values to a range between 0 and 20
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
        
        
        # Compute the standard error of the mean
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition']=='AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition']=='CO2']
        
        # Set the plot style
        sns.set(style='whitegrid')
        
        # Plot the line plot
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
            
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_air[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_air[feature_column], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column]-sem_gsr_co2[feature_column], 
                          mean_gsr[feature_column]+sem_gsr_co2[feature_column], alpha=0.2)
        
        
        # CO2 Condition FIT
        # Fit the 4th-order polynomial function to the data with adjusted initial guess
        x_co2 = windowed_features['window_index'][windowed_features['Condition'] == 'CO2']
        y_co2 = windowed_features[feature_column][windowed_features['Condition'] == 'CO2']
        
        # Initial guess values for 4th-order polynomial function parameters
        initial_guess_co2 = [1, 1, 1, 1, 1, 1]  # You may need to adjust these
        
        bounds_co2 = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, p0=initial_guess_co2, bounds=bounds_co2)
        
        # Extract the fitted parameters
        a, b, c, d, e, f = popt_co2
        
        # Print the 4th-order polynomial equation
        print(f"CO2 5th-Order Polynomial Equation: y = {a} * x^5 + {b} * x^4 + {c} * x^3 + {d} * x^2 + {e} * x + {f}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_co2 = np.linspace(min(x_co2), max(x_co2), 1000)  # Adjust the number of points as needed
        fit_line_co2 = co2_function(x_fit_co2, *popt_co2)
        
        # Scale x_fit_co2 to the range [0, 20] (if needed)
        x_fit_scaled_co2 = (x_fit_co2 - min(x_fit_co2)) / (max(x_fit_co2) - min(x_fit_co2)) * 20
        
        # Plot the 4th-order polynomial curve fit
        plt.plot(x_fit_scaled_co2, fit_line_co2, '--', color='red', label='5th-Order Polynomial Fit', linewidth=3)
        
        
        # AIR Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_air = windowed_features['window_index'][windowed_features['Condition'] == 'AIR']
        y_air = windowed_features[feature_column][windowed_features['Condition'] == 'AIR']
        popt_air, _ = curve_fit(air_function, x_air, y_air, bounds=([0, 0, 0], [np.max(y_air), np.max(x_air), np.max(x_air)]))
        
        a, b, c = popt_air
        # Print the logistic equation
        print(f"AIR Quadratic Equation: y = {a} * (x - {c})^2 + {b}")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_air = np.linspace(min(x_air), max(x_air), 1000)  # Adjust the number of points as needed
        fit_line_air = air_function(x_fit_air, *popt_air)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_air = (x_fit_air - min(x_fit_air)) / (max(x_fit_air) - min(x_fit_air)) * 20
        
        # Plot only the logistic curve fit
        plt.plot(x_fit_scaled_air, fit_line_air, '--', color='blue', label='Quadratic Fit', linewidth='3')
        
        
        
        # Customize plot titles and labels
        #plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.xticks(np.arange(0, 21, 5))
        plt.xlim([0, 20])
    
        
        #if(not plot_path):
          #   plt.savefig(plot_path)
        plt.savefig(plot_path)
        
        # Display the plot
        plt.show()
        
    def features_time_series_rr2(windowed_features, feature_column, title, xlabel, ylabel, air_function, co2_function, plot_path=False):
        plt.figure(figsize=(6, 5))
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].mean().reset_index()
    
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount() / mean_gsr['window_index'].max()) * 20
    
        sem_gsr = windowed_features.groupby(['Condition', 'window_index'])[feature_column].sem().reset_index()
        sem_gsr_air = sem_gsr[sem_gsr['Condition'] == 'AIR']
        sem_gsr_co2 = sem_gsr[sem_gsr['Condition'] == 'CO2']
    
        sns.set(style='whitegrid')
    
        sns.lineplot(x=mean_gsr['condition_index'], y=feature_column, hue='Condition', data=mean_gsr, linewidth=3)
    
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column] - sem_gsr_air[feature_column],
                         mean_gsr[feature_column] + sem_gsr_air[feature_column], alpha=0.2)
    
        plt.fill_between(mean_gsr['condition_index'], mean_gsr[feature_column] - sem_gsr_co2[feature_column],
                         mean_gsr[feature_column] + sem_gsr_co2[feature_column], alpha=0.2)
    
        # CO2 Condition FIT
        # Fit the logistic function to the data with adjusted bounds
        x_co2 = windowed_features['window_index'][windowed_features['Condition'] == 'CO2']
        y_co2 = windowed_features[feature_column][windowed_features['Condition'] == 'CO2']
        popt_co2, _ = curve_fit(co2_function, x_co2, y_co2, bounds=([0, 0, 0], [np.max(y_co2), np.max(x_co2), np.max(x_co2)]))
        
        a, b, c = popt_co2
        # Print the logistic equation
        print(f"CO2 Logistic Equation: y = {a} / (1 + e^(-{b} * (x - {c})))")
        
        # Generate y values using the fitted parameters for the desired x range
        x_fit_co2 = np.linspace(min(x_co2), max(x_co2), 1000)  # Adjust the number of points as needed
        fit_line_co2 = co2_function(x_fit_co2, *popt_co2)
        
        # Scale x_fit to the range [0, 20]
        x_fit_scaled_co2 = (x_fit_co2 - min(x_fit_co2)) / (max(x_fit_co2) - min(x_fit_co2)) * 20
        
        # Plot only the logistic curve fit
        plt.plot(x_fit_scaled_co2, fit_line_co2, '--', color='red', label='Logistic Fit', linewidth='3')
    
        # AIR Condition FIT
        x_air = windowed_features['window_index'][windowed_features['Condition'] == 'AIR']
        y_air = windowed_features[feature_column][windowed_features['Condition'] == 'AIR']
        popt_air, _ = curve_fit(air_function, x_air, y_air, bounds=([0, 0], [np.max(y_air), np.max(x_air)]))
    
        m_air, c_air = popt_air
        print(f"AIR Straight Line Equation: y = {m_air} * x + {c_air}")
    
        x_fit_air = np.linspace(min(x_air), max(x_air), 1000)
        fit_line_air = air_function(x_fit_air, *popt_air)
    
        x_fit_scaled_air = (x_fit_air - min(x_fit_air)) / (max(x_fit_air) - min(x_fit_air)) * 20
        plt.plot(x_fit_scaled_air, fit_line_air, '--', color='blue', label='Straight Line Fit', linewidth=3)
    
        plt.title(title, weight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim([0, 20])
    
        if plot_path:
            plt.savefig(plot_path)
        plt.show()
        
    def contact_gsr_line_plot(windowed_features, column_name, title, plot_directory):
        mean_contact = windowed_features.groupby(['Condition', 'window_index'])[column_name].mean().reset_index()
        mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
        mean_gsr = windowed_features.groupby(['Condition', 'window_index'])['Biopac_GSR_mean'].mean().reset_index()
        mean_gsr['condition_index'] = (mean_gsr.groupby('Condition').cumcount()/ mean_gsr['window_index'].max()) * 20
        
        
        plt.figure(figsize=(6, 5))
        
        # Compute SE of GSR
        gsr_se = windowed_features.groupby(['Condition', 'window_index'])['Biopac_GSR_mean'].sem().reset_index()
        gsr_se_air = gsr_se[gsr_se['Condition']=='AIR']
        gsr_se_co2 = gsr_se[gsr_se['Condition']=='CO2']
            
    
        # Plot SE of GSR
        plt.fill_between(mean_gsr['condition_index'], mean_gsr['Biopac_GSR_mean']-gsr_se_air['Biopac_GSR_mean'], 
                          mean_gsr['Biopac_GSR_mean']+gsr_se_air['Biopac_GSR_mean'], alpha=0.2)
        
        plt.fill_between(mean_gsr['condition_index'], mean_gsr['Biopac_GSR_mean']-gsr_se_co2['Biopac_GSR_mean'], 
                          mean_gsr['Biopac_GSR_mean']+gsr_se_co2['Biopac_GSR_mean'], alpha=0.2)
        
        
        # Compute SE of Contact column
        contact_se = windowed_features.groupby(['Condition', 'window_index'])[column_name].sem().reset_index()
        contact_se_air = contact_se[contact_se['Condition']=='AIR']
        contact_se_co2 = contact_se[contact_se['Condition']=='CO2']
            
        # Plot SE of Contact column
        plt.fill_between(mean_contact['condition_index'], mean_contact[column_name]-contact_se_air[column_name], 
                          mean_contact[column_name]+contact_se_air[column_name], alpha=0.2)
        
        plt.fill_between(mean_contact['condition_index'], mean_contact[column_name]-contact_se_co2[column_name], 
                          mean_contact[column_name]+contact_se_co2[column_name], alpha=0.2)
        
        # Set the plot style
        sns.set(style='whitegrid')
    
        # Plot the line plot
        sns.lineplot(x=mean_contact['condition_index'], y=column_name, hue='Condition', data=mean_contact, linewidth=3)
        sns.lineplot(x=mean_gsr['condition_index'], y='Biopac_GSR_mean', hue='Condition', data=mean_gsr, linestyle='--', alpha=0.5, linewidth=3)
    
        # Get the current axes
        ax = plt.gca()
    
        # Create custom legend handles and labels
        solid_line = mlines.Line2D([], [], color='blue', linestyle='-', label='AIR_Contact')
        dotted_line = mlines.Line2D([], [], color='blue', linestyle='--', label='AIR_GSR')
        solid_line2 = mlines.Line2D([], [], color='orange', linestyle='-', label='CO2_Contact')
        dotted_line2 = mlines.Line2D([], [], color='orange', linestyle='--', label='CO2_GSR')
    
        # Set the legend handles and labels
        handles = [solid_line, dotted_line, solid_line2, dotted_line2]
        labels = ['AIR_Contact', 'AIR_GSR', 'CO2_Contact', 'CO2_GSR']
        ax.legend(handles, labels)
    
        # Customize plot titles and labels
        plt.title(title, weight='bold')
        plt.xlabel('Time windows')
        plt.ylabel(column_name)
        
        plt.savefig(os.path.join(plot_directory, column_name.replace('/', '_').replace('[', '_').replace(']', '_')))
    
        # Display the plot
        plt.show()
        
    def segment_violin(segment_features, column_name, title, x_label, y_label, plot_directory):
            
        plt.figure(figsize=(6, 5))
        # Create the violin plot
        sns.violinplot(x='Condition', y=column_name, data=segment_features)
        
        
        # Customize plot titles and labels
        sns.set(style='whitegrid')
        #plt.title(title, weight='bold')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.figure(figsize=(16, 5))

        
        plt.savefig(os.path.join(plot_directory, column_name.replace('/', '_').replace('[', '_').replace(']', '_') + '_segments'))
        
        # Display the plot
        plt.show()
    
    def segment_violin_imu2(segment_features, column_name, title, x_label, y_label, plot_directory):
            
        # Create the violin plot
        sns.violinplot(x='Condition', y=column_name, data=segment_features)
        
        # Customize plot titles and labels
        sns.set(style='whitegrid')
        plt.title(title, weight='bold')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.figure(figsize=(10, 6))
        
        plt.savefig(os.path.join(plot_directory, column_name.replace('/', '_').replace('[', '_').replace(']', '_') + '_segments'))
        
        # Display the plot
        plt.show()
        
    def segment_violin_imu(segment_features, axes_columns, title, x_label, y_label, plot_directory):
        # Melt the DataFrame to create a single column for axes and a 'Condition' column
        melted_data = segment_features.melt(id_vars='Condition', value_vars=axes_columns, var_name='Axis', value_name='Value')
    
        # Create the violin plot
        violin = sns.violinplot(x='Condition', y='Value', hue='Axis', data=melted_data)
    
        # Customize plot titles and labels
        sns.set(style='whitegrid')
        plt.title(title, weight='bold')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Rename the legend
        legend = violin.get_legend()
        # Manually set labels
        legend.texts[0].set_text('X')
        legend.texts[1].set_text('Y')
        legend.texts[2].set_text('Z')
    
        # Save the plot
        plt.savefig(os.path.join(plot_directory, title.replace(' ', '_') + '_segments'))
    
        # Display the plot
        plt.show()
        
    def correlation_heatmap(data, plot_directory):
    
        data = data.sort_index()
        data = data.sort_index(axis=1)
        plt.figure(figsize=(12,12))
        sns.heatmap(data, annot=False, cmap='coolwarm', square=False)
        plt.tight_layout()
        plt.savefig(plot_directory)
        
    def contact_muscles(data, plot_path):
    
        muscles = ['LeftOrbicularis', 'RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']
        lines = []
        plt.figure(figsize=(10, 6))
        for muscle in muscles:
            muscle_name = f'Emg_Contact_{muscle}_mean'
            plot_title = f'{muscle} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[muscle_name].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[muscle_name], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, muscles)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Contact', weight='bold')
        plt.savefig(plot_path)
        plt.show()

    def contact_muscles2(air_data, co2_data, plot_path):
        muscles = ['LeftOrbicularis', 'RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)
        #fig.subplots_adjust(hspace=0.5)
        
        datasets = [(air_data, 'Air'), (co2_data, 'CO2')]
        
        for ax, (data, condition) in zip(axes, datasets):
            lines = []
            for muscle in muscles:
                muscle_name = f'Emg_Contact_{muscle}_mean'
                
                mean_contact = data.groupby(['window_index'])[muscle_name].mean().reset_index()
                mean_contact['condition_index'] = (mean_contact['window_index'] / mean_contact['window_index'].max()) * 20
                line, = ax.plot(mean_contact['condition_index'], mean_contact[muscle_name], linewidth=5, label=muscle)
                lines.append(line)
            
            ax.set_title(f'{condition} Condition', weight='bold')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Mean Skin Impedance')
            
            if condition == 'Air':
                ax.legend(lines, muscles, loc='upper right', prop={'weight':'bold'})  # Place legend in the top left corner of CO2 plot
                ax.set_ylim(0.14, 0.70)  # Set y-limits
            #else:
                #ax.legend(lines, muscles, loc='upper right')  # Use shared legend for the air plot
            
            #ax.set_ylim(0, 0.60)  # Set y-limits
    
        plt.savefig(plot_path)
        plt.show()
    
    def amp_muscles(data, plot_path):
    
        muscles = ['LeftOrbicularis', 'RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']
        lines = []
        plt.figure(figsize=(10, 6))
        for muscle in muscles:
            muscle_name = f'Emg_Amplitude_{muscle}_mean'
            plot_title = f'{muscle} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[muscle_name].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[muscle_name], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, muscles)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Amplitude', weight='bold')
        plt.savefig(plot_path)
        plt.show()
        
    def imu_gyro(data, plot_path):
    
        gyro_axis = ['Gyroscope_Raw.x_mean', 'Gyroscope_Raw.y_mean', 'Gyroscope_Raw.z_mean']
        lines = []
        plt.figure(figsize=(10, 6))
        for axis in gyro_axis:
            plot_title = f'{gyro_axis} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[axis].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[axis], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, gyro_axis)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Gyroscope', weight='bold')
        plt.savefig(plot_path)
        plt.show()
        
    def imu_acc(data, plot_path):
    
        gyro_axis = ['Accelerometer_Raw.x_mean', 'Accelerometer_Raw.y_mean', 'Accelerometer_Raw.z_mean']
        lines = []
        plt.figure(figsize=(10, 6))
        for axis in gyro_axis:
            plot_title = f'{gyro_axis} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[axis].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[axis], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, gyro_axis)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Accelerometer', weight='bold')
        plt.savefig(plot_path)
        plt.show()
        
    def contact_barplot(contact_air_windows, contact_co2_windows, plot_path):

        # Assuming you have already imported and prepared your dataframes
        # contact_air_windows and contact_co2_windows
        
        # Choose the columns representing different muscles/contacts to compare
        columns_to_compare = ['Emg/Contact[RightOrbicularis]_mean', 'Emg/Contact[RightZygomaticus]_mean', 'Emg/Contact[RightFrontalis]_mean', 'Emg/Contact[CenterCorrugator]_mean', 'Emg/Contact[LeftFrontalis]_mean', 'Emg/Contact[LeftZygomaticus]_mean', 'Emg/Contact[LeftOrbicularis]_mean']
        
        # Create a subset of the dataframes for the selected columns
        air_subset = contact_air_windows[columns_to_compare]
        co2_subset = contact_co2_windows[columns_to_compare]
        
        # Set up the figure and axes
        plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        
        # Width of the bars
        bar_width = 0.4
        
        # Positions for the bar groups
        positions_air = range(len(columns_to_compare))
        positions_co2 = [pos + bar_width for pos in positions_air]
        
        # Create the bar groups
        bars_air = ax.bar(positions_air, air_subset.mean(), width=bar_width, label='Air')
        bars_co2 = ax.bar(positions_co2, co2_subset.mean(), width=bar_width, label='CO2')
        
        # Set x-axis labels and ticks
        ax.set_xticks([pos + bar_width / 2 for pos in positions_air])
        ax.set_xticklabels([col.split('[')[1].split(']')[0] for col in columns_to_compare])
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis label
        ax.set_ylabel('Mean Skin Impedance')
        
        # Set plot title and legend
        plt.title('Mean Skin Impedance Comparison between Air and CO2')
        plt.legend()
        
        # Show the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        
    def contact_barplot_with_se(contact_air_windows, contact_co2_windows, plot_path):
    
        # Assuming you have already imported and prepared your dataframes
        # contact_air_windows and contact_co2_windows
        
        # Choose the columns representing different muscles/contacts to compare
        #columns_to_compare = ['Emg_Contact_RightOrbicularis_mean', 'Emg_Contact_RightZygomaticus_mean', 'Emg_Contact_RightFrontalis_mean', 'Emg_Contact_CenterCorrugator_mean', 'Emg_Contact_LeftFrontalis_mean', 'Emg_Contact_LeftZygomaticus_mean', 'Emg_Contact_LeftOrbicularis_mean']
        columns_to_compare = ['Emg_Contact_LeftFrontalis_mean', 'Emg_Contact_RightFrontalis_mean', 'Emg_Contact_LeftOrbicularis_mean', 'Emg_Contact_RightOrbicularis_mean', 'Emg_Contact_LeftZygomaticus_mean', 'Emg_Contact_RightZygomaticus_mean', 'Emg_Contact_CenterCorrugator_mean']
        
        # Create a subset of the dataframes for the selected columns
        air_subset = contact_air_windows[columns_to_compare]
        co2_subset = contact_co2_windows[columns_to_compare]
        
        # Set up the figure and axes
        plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        
        # Width of the bars
        bar_width = 0.4
        
        # Positions for the bar groups
        positions_air = np.arange(len(columns_to_compare))
        positions_co2 = positions_air + bar_width
        
        # Compute mean and standard error for each group
        mean_air = air_subset.mean()
        sem_air = air_subset.sem()
        mean_co2 = co2_subset.mean()
        sem_co2 = co2_subset.sem()
        
        # Create the bar groups with error bars
        bars_air = ax.bar(positions_air, mean_air, yerr=sem_air, width=bar_width, label='Air', capsize=5)
        bars_co2 = ax.bar(positions_co2, mean_co2, yerr=sem_co2, width=bar_width, label='CO2', capsize=5)
        
        # Set x-axis labels and ticks
        ax.set_xticks(positions_air + bar_width / 2 + 0.4)
        #ax.set_xticklabels(['RightOrbicularis', 'RightZygomaticus', 'RightFrontalis', 'CenterCorrugator', 'LeftFrontalis', 'LeftZygomaticus', 'LeftOrbicularis'])
        ax.set_xticklabels(['LeftFrontalis*', 'RightFrontalis', 'LeftOrbicularis*', 'RightOrbicularis*', 'RightZygomaticus*', 'LeftZygomaticus*', 'CenterCorrugator*'])
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis label
        ax.set_ylabel('Facial Skin Impedance')
        
        # Set plot title and legend
        #plt.title('Mean Skin Impedance Comparison between Air and CO2')
        plt.legend()
        
        # Show the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        
    def amp_barplot_with_se(amp_air_windows, amp_co2_windows, plot_path):
    
        # Assuming you have already imported and prepared your dataframes
        # contact_air_windows and contact_co2_windows
        
        # Choose the columns representing different muscles/contacts to compare
        #columns_to_compare = ['Emg_Amplitude_RightOrbicularis_mean', 'Emg_Amplitude_RightZygomaticus_mean', 'Emg_Amplitude_RightFrontalis_mean', 'Emg_Amplitude_CenterCorrugator_mean', 'Emg_Amplitude_LeftFrontalis_mean', 'Emg_Amplitude_LeftZygomaticus_mean', 'Emg_Amplitude_LeftOrbicularis_mean']
        columns_to_compare = ['Emg_Amplitude_LeftFrontalis_mean', 'Emg_Amplitude_RightFrontalis_mean', 'Emg_Amplitude_LeftOrbicularis_mean', 'Emg_Amplitude_RightOrbicularis_mean', 'Emg_Amplitude_LeftZygomaticus_mean', 'Emg_Amplitude_RightZygomaticus_mean', 'Emg_Amplitude_CenterCorrugator_mean']
        
        # Create a subset of the dataframes for the selected columns
        air_subset = amp_air_windows[columns_to_compare]
        co2_subset = amp_co2_windows[columns_to_compare]
        
        # Set up the figure and axes
        plt.figure(figsize=(8, 6))
        ax = plt.subplot()
        
        # Width of the bars
        bar_width = 0.4
        
        # Positions for the bar groups
        positions_air = np.arange(len(columns_to_compare))
        positions_co2 = positions_air + bar_width
        
        # Compute mean and standard error for each group
        mean_air = air_subset.mean()
        sem_air = air_subset.sem()
        mean_co2 = co2_subset.mean()
        sem_co2 = co2_subset.sem()
        
        # Create the bar groups with error bars
        bars_air = ax.bar(positions_air, mean_air, yerr=sem_air, width=bar_width, label='Air', capsize=5)
        bars_co2 = ax.bar(positions_co2, mean_co2, yerr=sem_co2, width=bar_width, label='CO2', capsize=5)
        
        # Set x-axis labels and ticks
        ax.set_xticks(positions_air + bar_width / 2 + 0.4)
        #ax.set_xticklabels(['RightOrbicularis', 'RightZygomaticus', 'RightFrontalis', 'CenterCorrugator', 'LeftFrontalis', 'LeftZygomaticus', 'LeftOrbicularis'])
        ax.set_xticklabels(['LeftFrontalis', 'RightFrontalis', 'LeftOrbicularis*', 'RightOrbicularis*', 'RightZygomaticus*', 'LeftZygomaticus*', 'CenterCorrugator'])
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis label
        ax.set_ylabel('Facial EMG Amplitude')
        
        # Set plot title and legend
        #plt.title('Mean EMG Amplitude Comparison between Air and CO2')
        plt.legend()
        
        # Show the plot
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()
        
    def ratings_barplot(barplot_data, plot_path):

        # Convert data to numeric, ignoring errors
        baseline_data = barplot_data[['baseline_arousal_1', 'baseline_valence_1', 'baseline_anxiety']].apply(pd.to_numeric, errors='coerce').dropna()
        category_1_data = barplot_data[['arousal_1_1', 'valence_1_1', 'anxiety_1']].apply(pd.to_numeric, errors='coerce').dropna()
        category_2_data = barplot_data[['arousal_2_1', 'valence_2_1', 'anxiety_2']].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Plotting
        categories = ['Baseline', 'Air', 'CO2']
        variables = ['Arousal', 'Valence', 'Anxiety']
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 3 columns
        
        colors = [(0/255, 166/255, 147/255), (102/255, 110/255, 169/255), (190/255, 142/255, 93/255)]
        
        for i, ax in enumerate(axs):
            data = [baseline_data.iloc[:, i], category_1_data.iloc[:, i], category_2_data.iloc[:, i]]
            ax.bar(categories, [np.mean(d) for d in data], yerr=[np.std(d) / np.sqrt(len(d)) for d in data], capsize=5, color=colors)
            ax.set_ylabel(variables[i])
            ax.set_title(variables[i])
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()



    
    
    
    
    
    
