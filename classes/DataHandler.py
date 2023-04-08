# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:32:13 2023

@author: Michal Gnacek (www.gnacek.com)
"""

import pandas as pd
import os
import numpy as np

from utils.load_data import load_data_with_event_matching
from utils.timestamps import j2000_to_unix, generate_biopac_unix_timestamps
import utils.constants as constants

class DataHandler:
    
    def load_mask_data(mask_df, mask_event_df, participant_id):
        data = pd.DataFrame()
        if os.path.exists(mask_df):
            if os.path.exists(mask_event_df):
                print('Loading data for participant: ' + participant_id)
                data = load_data_with_event_matching(mask_df, True, mask_event_df)
                data = data.drop(columns=constants.REDUNDANT_MASK_COLUMNS)
                print('Finished loading data for participant: ' + participant_id)
            else:
                print("Event file missing for participant: " + participant_id)
        else:
            print("Mask file missing for participant: " + participant_id)
        return data
            
    def load_eyetracking_data(eye_df, participant_id):
        # Load eye tracking data
        data = pd.DataFrame()
        
        if os.path.exists(eye_df):
            print('Loading eye data for participant: ' + participant_id)
            data = pd.read_csv(eye_df)
            print('Finished loading eye data for participant: ' + participant_id)
            # First couple frames of eye tracking data are weird. Drop them.
            data = data.drop(index=range(10)).reset_index(drop=True)
        else:
            print("Eye file missing for participant: " + participant_id)
        return data
    
    def load_biopac_data(biopac_df, biopac_start_unix, participant_id):
        BIOPAC_COLUMNS_NAMES = ['Biopac_GSR', 'Biopac_RSP']
        # Load biopac data
        data = pd.DataFrame()
        
        if os.path.exists(biopac_df) and biopac_start_unix:
            print('Loading biopac data for participant: ' + participant_id)
            data = pd.read_csv(biopac_df, names=BIOPAC_COLUMNS_NAMES, index_col=False)
            # Generate biopac unix timestamps at 1000hz from the start of the file
            biopac_unix_timestamps = generate_biopac_unix_timestamps(data, biopac_start_unix)
            data = pd.concat([data,biopac_unix_timestamps], axis=1)
        return data
    
    def sync_signal_data(mask_df, eye_df, biopac_df, biopac_start_unix):
        synced_data = mask_df.copy()
        # Add eye tracking data

        # EYE TRACKING SYNC: 

        if mask_df.empty or eye_df.empty:
            print('Cant synchronise eye tracking data. One of the files is empty')
        else:
            #Some eye tracking files (participants 1-9) have empty 'TimestampUnix column'. Use their 'TimestampJ2000' to calculate unix column
            if np.mean(eye_df['TimestampUnix'].values)==0:
                print('TimestampUnix column for eye data is empty. Converting J2000 timestamps to Unix')
                j2000_timestamps = eye_df['TimestampJ2000']
                unix_timestamps = []
                for j2000_timestamp in j2000_timestamps:
                    unix_timestamps.append(j2000_to_unix(j2000_timestamp))
                eye_df['TimestampUnix']=unix_timestamps
                
            eye_df['TimestampUnix'] = eye_df['TimestampUnix']/1000
                
            # SYNCHRONISE EYE DATA WITH MASK DF
            print('Synchronising mask and eye tracking files.')
            eye_tracking_columns = ['VerboseData.Right.PupilDiameterMm', 'VerboseData.Left.PupilDiameterMm']
            synced_eye_df = pd.DataFrame(np.nan, index=mask_df.index, columns=eye_tracking_columns)
            for index, eye_data_row in eye_df.iterrows():
                timestamp_to_find = eye_data_row['TimestampUnix']
                event_row_index = mask_df['unix_timestamp'].searchsorted(timestamp_to_find)
                synced_eye_df.loc[event_row_index] = eye_data_row[eye_tracking_columns]
            # eye data file sometimes runs for a few seconds longer than the mask data. 
            # Sync function above contiounsly overwrites last row of data with those eye data frames. Set the last row back to 0 for forward propagation later.     
            synced_eye_df.loc[len(synced_eye_df)-1]=np.nan
            synced_data[eye_tracking_columns] = synced_eye_df
            print('Forward filling eye tracking data')
            synced_data[eye_tracking_columns] = synced_data[eye_tracking_columns].ffill()
            print('Finished synchronising mask and eye tracking files')
        
        if mask_df.empty or eye_df.empty:
            print('Cant synchronise biopac data. One of the files is empty')
        else:
            print('Synchronising mask and biopac files.')
            BIOPAC_COLUMNS_NAMES = ['Biopac_GSR', 'Biopac_RSP']
            synced_data[BIOPAC_COLUMNS_NAMES] = np.nan
            if(biopac_start_unix>synced_data.loc[0]['unix_timestamp']):
                #Mask data started recording first
                #Find first matching frame of biopac data in mask
                first_row_index = synced_data['unix_timestamp'].searchsorted(biopac_df.loc[0]['unix_timestamp'])
                # Since mask data started first, first n of rows in mask data will not have biopac data. Fill them with nans
                nan_rows = pd.DataFrame(np.nan, index=np.arange(first_row_index), columns=BIOPAC_COLUMNS_NAMES)
                biopac_df = pd.concat([nan_rows, biopac_df]).reset_index(drop=True)
            else:
                #Biopac data started recording first
                #Find first matching frame of mask data in biopac
                first_row_index = biopac_df['unix_timestamp'].searchsorted(synced_data.loc[0]['unix_timestamp'])
                #Since biopac started first, cant sync first n number of rows. Based on first shared timestamp, drop biopac data before mask recording started
                biopac_df = biopac_df.loc[first_row_index:].reset_index(drop=True)
                #now that both biopac and mask df start at the same time, trim biopac data to end when mask data ends based on the same frequency
                biopac_df = biopac_df.loc[:len(synced_data)-1].reset_index(drop=True)

            # merge biopac and mask df
            synced_data[BIOPAC_COLUMNS_NAMES] = biopac_df[BIOPAC_COLUMNS_NAMES]
            print('Finished synchronising mask and biopac files')
            
        return synced_data
    



    




  