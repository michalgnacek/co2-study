# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:07:08 2023

@author: Michal Gnacek (www.gnacek.com)
"""
import os
from utils.timestamps import biopac_file_name_to_unix

class Participant:
    def __init__(self, folder):
        self.folder = folder
        self.id = folder.split("\\")[len(folder.split("\\"))-1]
        
        # Initialize variables for air condition
        air_dir = os.path.join(folder, "air")
        
        self.air_mask_file = None
        self.air_eye_file = None
        self.air_event_file = None
        self.air_biopac_file = None
        self.air_biopac_start_time_file = None
        self.air_biopac_unix_start_time = None
        
        # Initialize variables for co2 condition
        co2_dir = os.path.join(folder, "co2")
        
        self.co2_mask_file = None
        self.co2_eye_file = None
        self.co2_event_file = None
        self.co2_biopac_file = None
        self.co2_biopac_start_time_file = None
        self.co2_biopac_unix_start_time = None
    
        if(os.path.exists(air_dir)):
            for file in os.listdir(air_dir):
                if file.endswith(".csv"):
                    if "T" in file:
                        if "t" not in file:
                            self.air_mask_file = os.path.join(air_dir, file)
                    elif "eyedata" in file:
                        self.air_eye_file = os.path.join(air_dir, file)
                elif file.endswith(".json"):
                    self.air_event_file = os.path.join(air_dir, file)
                elif file.endswith(".txt"):
                    if "air.txt" in file:
                        self.air_biopac_file = os.path.join(air_dir, file)
                    elif "2022" in file:
                        self.air_biopac_start_time_file = os.path.join(air_dir, file)
                        self.air_biopac_unix_start_time = biopac_file_name_to_unix(file)
                        
                    
        if(os.path.exists(co2_dir)):
            for file in os.listdir(co2_dir):
                if file.endswith(".csv"):
                    if "T" in file:
                        if "t" not in file:
                            self.co2_mask_file = os.path.join(co2_dir, file)
                    elif "eyedata" in file:
                        self.co2_eye_file = os.path.join(co2_dir, file)
                elif file.endswith(".json"):
                    self.co2_event_file = os.path.join(co2_dir, file)
                elif file.endswith(".txt"):
                    if "co2.txt" in file:
                        self.co2_biopac_file = os.path.join(co2_dir, file)
                    elif "2022" in file:
                        self.co2_biopac_start_time_file = os.path.join(air_dir, file)
                        self.co2_biopac_unix_start_time = biopac_file_name_to_unix(file)
        
