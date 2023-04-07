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
    utc_date_time = datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(milliseconds=j2000_timestamp)
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
    utc_date_time = datetime.datetime(2000, 1, 1, 1, 0) + datetime.timedelta(milliseconds=j2000_timestamp)
    unix_timestamp = int(utc_date_time.timestamp()*1000)
    return unix_timestamp
