# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:14:33 2023

@author: m
"""

from enum import Enum

class AirFiles(Enum):
    MASK = 'air_mask_file'
    EYE = 'air_eye_file'
    EVENT = 'air_event_file'
    BIOPAC = 'air_biopac_file'
    BIOPAC_TIME = 'air_biopac_start_time_file'

class CO2Files(Enum):
    MASK = 'co2_mask_file'
    EYE = 'co2_eye_file'
    EVENT = 'co2_event_file'
    BIOPAC = 'co2_biopac_file'
    BIOPAC_TIME = 'co2_biopac_start_time_file'