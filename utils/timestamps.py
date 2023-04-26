# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : 
#           Michal Gnacek | gnacek.com
# Created Date: 07/04/2023
# =============================================================================
"""
Utility functions to deal with timestamp management
"""
# =============================================================================
# Imports
# =============================================================================
import datetime
import pandas as pd

# =============================================================================
# Main
# =============================================================================

    
def read_unix(unix_timestamp):
    """
    
    Parameters
    ----------
    unix_timestamp : int
        UNIX TIMESTAMP TO BE CONVERTED.

    Returns
    -------
    formatted_date_time : String
        HUMAN READABLE DATATIME OBJECT OF A CONVERTED UNIX TIMESTAMP.

    """
    unix_timestamp = unix_timestamp
    utc_date_time = datetime.datetime.utcfromtimestamp(unix_timestamp)
    formatted_date_time = utc_date_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    return formatted_date_time

def read_j2000(j2000_timestamp):
    """
    
    Parameters
    ----------
    unix_timestamp : int
        J2000 TIMESTAMP TO BE CONVERTED.

    Returns
    -------
    formatted_date_time : String
        HUMAN READABLE DATATIME OBJECT OF A CONVERTED J2000 TIMESTAMP.

    """
    j2000_timestamp = 704397173515
    utc_date_time = datetime.datetime(2000, 1, 1, 1, 0) + datetime.timedelta(milliseconds=j2000_timestamp)
    formatted_date_time = utc_date_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    return formatted_date_time
    
def j2000_to_unix(j2000_timestamp):
    """

    Parameters
    ----------
    j2000_timestamp : int
        J2000 TIMESTAMP TO BE CONVERTED.

    Returns
    -------
    unix_timestamp : int
        CONVERTED UNIX TIMESTAMP.

    """
    utc_date_time = datetime.datetime(2000, 1, 1, 3, 0) + datetime.timedelta(milliseconds=j2000_timestamp)
    unix_timestamp = int(utc_date_time.timestamp()*1000)
    return unix_timestamp

def biopac_file_name_to_unix(biopac_file_name):
    """

    Parameters
    ----------
    j2000_timestamp : Str
        Text file name of a file indicating biopac start time.

    Returns
    -------
    unix_timestamp : int
        UNIX TIMESTAMP FOR START OF BIOPAC RECORDING.

    """
    if('air' in biopac_file_name):
        pattern = "air_%d_%m_%Y_%H_%M_%S.txt"
    elif('co2' in biopac_file_name):
        pattern = "co2_%d_%m_%Y_%H_%M_%S.txt"
    
    else:
        print('Incorrect biopac time file name. Unable to synchronise biopac and mask data.')
    
    return (datetime.datetime.strptime(biopac_file_name, pattern)).timestamp()+7200

def generate_biopac_unix_timestamps(biopac_df, biopac_start_unix):
    """

    Parameters
    ----------
    biopac_df : dataframe
        PANDAS DATAFRAME HOLDING BIOPAC DATA.
    biopac_start_unix : float
        UNIX TIMESTAMP INDICATING START OF THE BIOPAC RECORDING.

    Returns
    -------
    biopac_unix_timestamps : dataframe
        PANDAS DATAFRAME HOLDING A LIST OF UNIX TIMESTAMPS GENERATED FOR THE BIOPAC FILE AT 1000HZ STARTING FROM INDICATED START TIME.

    """

    date_range = pd.date_range(start=datetime.datetime.fromtimestamp(biopac_start_unix), periods=len(biopac_df), freq='1ms')
    biopac_unix_timestamps = pd.DataFrame([(datetime.datetime.timestamp(dt)) for dt in date_range], columns=['unix_timestamp'])
    return biopac_unix_timestamps