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
    BIOPAC_TIME_FILE = 'air_biopac_start_time_file'
    BIOPAC_UNIX_START_TIME = 'air_biopac_unix_start_time'

class CO2Files(Enum):
    MASK = 'co2_mask_file'
    EYE = 'co2_eye_file'
    EVENT = 'co2_event_file'
    BIOPAC = 'co2_biopac_file'
    BIOPAC_TIME_FILE = 'co2_biopac_start_time_file'
    BIOPAC_UNIX_START_TIME = 'co2_biopac_unix_start_time'
    
REDUNDANT_MASK_COLUMNS = ['Emg/RawLift[RightOrbicularis]', 'Emg/RawLift[RightZygomaticus]', 'Emg/RawLift[RightFrontalis]', 'Emg/RawLift[CenterCorrugator]', 'Emg/RawLift[LeftFrontalis]', 'Emg/RawLift[LeftZygomaticus]', 'Emg/RawLift[LeftOrbicularis]',
                          'Emg/ContactStates[RightOrbicularis]', 'Emg/ContactStates[RightZygomaticus]', 'Emg/ContactStates[RightFrontalis]', 'Emg/ContactStates[CenterCorrugator]', 'Emg/ContactStates[LeftFrontalis]', 'Emg/ContactStates[LeftZygomaticus]', 'Emg/ContactStates[LeftOrbicularis]',
                          'Magnetometer/Raw.x', 'Magnetometer/Raw.y', 'Magnetometer/Raw.z']

FIT_STATE_THRESHOLD = 7
FIT_STATE_THRESHOLD2 = 7

class DATA_COLUMNS(Enum):
    EYE_LEFT_PUPIL_SIZE = 'VerboseData.Left.PupilDiameterMm'
    EYE_RIGHT_PUPIL_SIZE = 'VerboseData.Right.PupilDiameterMm'
    
class FREQUENCIES(Enum):
    EYE_TRACKING = 120