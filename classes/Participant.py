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
