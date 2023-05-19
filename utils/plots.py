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

class Plots:

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
        
    def correlation_heatmap(data, plot_directory):
    
        plt.figure(figsize=(12,12))
        sns.heatmap(data, annot=False, cmap='coolwarm', square=False)
        plt.tight_layout()
        plt.savefig(plot_directory)
        
    def contact_muscles(data, plot_path):
    
        muscles = ['LeftOrbicularis', 'RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']
        lines = []
        plt.figure(figsize=(10, 6))
        for muscle in muscles:
            muscle_name = f'Emg/Contact[{muscle}]_mean'
            plot_title = f'{muscle} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[muscle_name].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[muscle_name], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, muscles)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Contact', weight='bold')
        plt.savefig(plot_path)
        plt.show()
    
    def amp_muscles(data, plot_path):
    
        muscles = ['LeftOrbicularis', 'RightOrbicularis', 'LeftFrontalis', 'RightFrontalis', 'LeftZygomaticus', 'RightZygomaticus', 'CenterCorrugator']
        lines = []
        plt.figure(figsize=(10, 6))
        for muscle in muscles:
            muscle_name = f'Emg/Amplitude[{muscle}]_mean'
            plot_title = f'{muscle} Mean Over Time'
            mean_contact = data.groupby(['Condition', 'window_index'])[muscle_name].mean().reset_index()
            mean_contact['condition_index'] = (mean_contact.groupby('Condition').cumcount() / mean_contact['window_index'].max()) * 20
            line, = plt.plot(mean_contact['condition_index'],mean_contact[muscle_name], linewidth=3)
            lines.append(line)
        
        plt.legend(lines, muscles)
        
        plt.title('Mean ' + mean_contact['Condition'].unique()[0] +' Amplitude', weight='bold')
        plt.savefig(plot_path)
        plt.show()
    
    
    
    
    
    
