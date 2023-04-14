# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:32:13 2023

@author: Michal Gnacek (www.gnacek.com)
"""

import pandas as pd

class Participant:
    def __init__(self, id):
        self.id = str(id)
        self.air_mask_data = pd.DataFrame()
        self.air_eye_data = pd.DataFrame()
        self.air_biopac_data = pd.DataFrame()
        self.air_synced_data = pd.DataFrame()
        
        self.co2_mask_data = pd.DataFrame()
        self.co2_eye_data = pd.DataFrame()
        self.co2_biopac_data = pd.DataFrame()
        self.co2_synced_data = pd.DataFrame()
        
    def set_air_mask_data(self, data):
        data.insert(0, 'Participant_No', self.id)
        data.insert(1, 'Condition', 'AIR')
        self.air_mask_data = data
        
    def set_air_eye_data(self, data):
        self.air_eye_data = data
        
    def set_air_biopac_data(self, data):
        self.air_biopac_data = data
        
    def set_air_synced_data(self, data):
        self.air_synced_data = data
        
    def set_co2_mask_data(self, data):
        data.insert(0, 'Participant_No', self.id)
        data.insert(1, 'Condition', 'CO2')
        self.co2_mask_data = data
        
    def set_co2_eye_data(self, data):
        self.co2_eye_data = data
        
    def set_co2_biopac_data(self, data):
        self.co2_biopac_data = data
        
    def set_co2_synced_data(self, data):
        self.co2_synced_data = data
    
    def get_expression_calibration_data(self, condition):
        if condition=='air':
            data = self.air_synced_data
        elif condition=='co2':
            data = self.co2_synced_data
        else:
            print('Invalid condition')
        # Get start and end unix timestamps for expression calibration
        calibration_data_row_pair = data[data['Event']=='Calibration']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for expression calibration')
        else:
            data = data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1]]
        return data
    
    def get_brightness_calibration_data(self, condition):
        if condition=='air':
            data = self.air_synced_data
        elif condition=='co2':
            data = self.co2_synced_data
        else:
            print('Invalid condition')
        # Get start and end unix timestamps for brightness calibration
        calibration_data_row_pair = data[data['Event']=='Brightness Calibration']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for expression calibration')
        else:
            data = data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1]]
        return data
    
    def get_condition_data(self, condition):
        if condition=='air':
            data = self.air_synced_data
        elif condition=='co2':
            data = self.co2_synced_data
        else:
            print('Invalid condition')
        # Get start and end unix timestamps for expression calibration
        calibration_data_row_pair = data[data['Event']=='Condition 1']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for expression calibration')
        else:
            data = data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1]]
        #participant.air_synced_data['Event'][participant.air_synced_data['Event']!='']
        return data
        
