# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 02:23:12 2023

@author: m
"""

import matplotlib.pyplot as plt
import os

def plot_eyetracking_filter(raw_signal, filtered_signal, participant_id, condition):
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
    
def plot_participant_overview(participant_df, save_plot):

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
   
    
    
    # Assume GSR_air, GSR_co2, BR_air, BR_co2 are NumPy arrays containing the signal values
    # Assume condition_1 and condition_2 are strings containing the names of the conditions
    
    # Set up the figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(7, 2, figsize=(10, 8))
    
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

