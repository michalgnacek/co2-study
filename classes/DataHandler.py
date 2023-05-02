# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:32:13 2023

@author: Michal Gnacek (www.gnacek.com)
"""

import pandas as pd
pd.options.mode.chained_assignment = None

import os
import numpy as np

from utils.load_data import load_data_with_event_matching
from utils.timestamps import j2000_to_unix, generate_biopac_unix_timestamps
from utils.impute_eye_tracking_data import impute_eye_data
import utils.constants as constants
from utils.normalisation import min_max_normalisation
from sklearn.preprocessing import MinMaxScaler
import utils.filters as filters
from utils.generate_sliding_windows import generate_sliding_windows
from classes.Features import calculate_statistical_features

class DataHandler:
    
    def load_mask_data(mask_df, mask_event_df, participant_id):
        data = pd.DataFrame()
        if os.path.exists(mask_df):
            if os.path.exists(mask_event_df):
                print('Loading mask data for participant: ' + participant_id)
                data = load_data_with_event_matching(mask_df, True, mask_event_df)
                data = data.drop(columns=constants.REDUNDANT_MASK_COLUMNS)
                print('Finished mask loading data for participant: ' + participant_id)
            else:
                print("Event file missing for participant: " + participant_id)
        else:
            print("Mask file missing for participant: " + participant_id)
        return data
                    
    def load_eyetracking_data(eye_df, participant_id, condition):
        if(os.path.getsize(eye_df)==0):
            return pd.DataFrame()
        else:
            processed_eye_data_directory = os.path.join(os.getcwd(), 'temp', 'processed_eye_data')
            participant_processed_eye_file = os.path.join(processed_eye_data_directory, participant_id + '_' + condition + '.csv')
            # Load eye tracking data
            print('Loading eye data for participant: ' + participant_id)
            data = pd.DataFrame()
            #if temp folder exists
            if(os.path.exists(processed_eye_data_directory)):
                #Eye tracking folder within temp folder exists, check for individual processed participant file
                if os.path.exists(participant_processed_eye_file):
                    #load filled eye tracking date
                    print('Imputed eye data file exists. Loading from temp/processed_eye_data')
                    data = pd.read_csv(participant_processed_eye_file, index_col=0)
                else:
                    data = impute_eye_data(eye_df)
                    data.to_csv(participant_processed_eye_file)
            else:
                os.mkdir(processed_eye_data_directory)
                data = impute_eye_data(eye_df)
                data.to_csv(participant_processed_eye_file)
            print('Finished loading eye data for participant: ' + participant_id)
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
        _eye_df = eye_df.copy()
        _biopac_df = biopac_df.copy()
        # Add eye tracking data

        # EYE TRACKING SYNC: 

        if mask_df.empty or _eye_df.empty:
            print('Cant synchronise eye tracking data. One of the files is empty')
        else:
            #Some eye tracking files (participants 1-9) have empty 'TimestampUnix column'. Use their 'TimestampJ2000' to calculate unix column
            if np.mean(_eye_df['TimestampUnix'].values)==0:
                # TimestampUnix column for eye data is empty. Converting J2000 timestamps to Unix
                j2000_timestamps = _eye_df['TimestampJ2000']
                unix_timestamps = []
                for j2000_timestamp in j2000_timestamps:
                    unix_timestamps.append(j2000_to_unix(j2000_timestamp))
                _eye_df['TimestampUnix']=unix_timestamps
                
            _eye_df['TimestampUnix'] = _eye_df['TimestampUnix']/1000
            
            # SYNCHRONISE EYE DATA WITH MASK DF
            print('Synchronising mask and eye tracking files.')
            eye_tracking_columns = ['VerboseData.Right.PupilDiameterMm', 'VerboseData.Left.PupilDiameterMm']
            synced_eye_df = pd.DataFrame(np.nan, index=mask_df.index, columns=eye_tracking_columns)
            for index, eye_data_row in _eye_df.iterrows():
                timestamp_to_find = eye_data_row['TimestampUnix']
                event_row_index = mask_df['unix_timestamp'].searchsorted(timestamp_to_find)
                synced_eye_df.loc[event_row_index] = eye_data_row[eye_tracking_columns]
            # eye data file sometimes runs for a few seconds longer than the mask data. 
            # Sync function above contiounsly overwrites last row of data with those eye data frames. Set the last row back to 0 for forward propagation later.     
            synced_eye_df.loc[len(synced_eye_df)-1]=np.nan
            synced_data[eye_tracking_columns] = synced_eye_df
            # Forward filling eye tracking data
            synced_data[eye_tracking_columns] = synced_data[eye_tracking_columns].ffill()
            print('Finished synchronising mask and eye tracking files')
        
        if mask_df.empty or _biopac_df.empty:
            print('Cant synchronise biopac data. One of the files is empty')
        else:
            print('Synchronising mask and biopac files.')
            BIOPAC_COLUMNS_NAMES = ['Biopac_GSR', 'Biopac_RSP']
            synced_data[BIOPAC_COLUMNS_NAMES] = np.nan
            if(biopac_start_unix>synced_data.loc[0]['unix_timestamp']):
                #Mask data started recording first
                #Find first matching frame of biopac data in mask
                first_row_index = synced_data['unix_timestamp'].searchsorted(_biopac_df.loc[0]['unix_timestamp'])
                # Since mask data started first, first n of rows in mask data will not have biopac data. Fill them with nans
                nan_rows = pd.DataFrame(np.nan, index=np.arange(first_row_index), columns=BIOPAC_COLUMNS_NAMES)
                _biopac_df = pd.concat([nan_rows, _biopac_df]).reset_index(drop=True)
            else:
                #Biopac data started recording first
                #Find first matching frame of mask data in biopac
                first_row_index = _biopac_df['unix_timestamp'].searchsorted(synced_data.loc[0]['unix_timestamp'])
                #Since biopac started first, cant sync first n number of rows. Based on first shared timestamp, drop biopac data before mask recording started
                _biopac_df = _biopac_df.loc[first_row_index:].reset_index(drop=True)
                #now that both biopac and mask df start at the same time, trim biopac data to end when mask data ends based on the same frequency
                _biopac_df = _biopac_df.loc[:len(synced_data)-1].reset_index(drop=True)

            # merge biopac and mask df
            synced_data[BIOPAC_COLUMNS_NAMES] = _biopac_df[BIOPAC_COLUMNS_NAMES]
            print('Finished synchronising mask and biopac files')
            
        return synced_data
    
    def downsample_participant_data(participant_id, air_df, co2_df):
                
        # concatenate all data frames into a single data frame
        combined_data = pd.concat([air_df, co2_df])
        
        # reset the index of the new data frame
        combined_data.reset_index(drop=True, inplace=True)
        
        df_downsampled = combined_data.iloc[::20]
        df_downsampled.reset_index(drop=True, inplace=True)
        
        processed_participant_directory = os.path.join(os.getcwd(), 'temp', 'synced_participant_data')
        #if temp folder does not exist
        if(not os.path.exists(processed_participant_directory)):
            os.mkdir(processed_participant_directory)
            
        processed_participant_file = os.path.join(processed_participant_directory, str(participant_id))
        df_downsampled.to_csv(processed_participant_file + '.csv', index=False)
        return df_downsampled

    def normalise_data_old(participant_df):
        # This normalises data for each participant for each condition
        data_to_normalise = participant_df.copy()
        #Remove data before calibration and after condition ended
        data_to_normalise = data_to_normalise[data_to_normalise['Segment'].notna()].reset_index(drop=True)
        normalised_data = data_to_normalise.copy()

        # Normalise data for each modality independently for each condition
        air_data = data_to_normalise.copy()[participant_df['Condition']=='AIR']
        co2_data = data_to_normalise.copy()[participant_df['Condition']=='CO2']

        
        for col_name, col_data in data_to_normalise.iteritems():
            #if column needs to be normalised
            if(col_name in constants.NORMALISATION_COLUMNS):
                air_data = col_data[data_to_normalise['Condition']=='AIR']
                co2_data = col_data[data_to_normalise['Condition']=='CO2']
                normalised_air_data = min_max_normalisation(air_data)
                normalised_co2_data = min_max_normalisation(co2_data)
                
                normalised_data[col_name][air_data.index[0]:air_data.index[len(normalised_air_data)-1]+1] = normalised_air_data.reshape(-1)
                normalised_data[col_name][co2_data.index[0]:co2_data.index[len(normalised_co2_data)-1]+1] = normalised_co2_data.reshape(-1)
                
        #Check that all segments and condition labels are prescent and that GSR and Eye tracking have been synced
        if (len(normalised_data['Condition'].unique())!=2):
            print('!!!Condition missing')
        if (len(normalised_data['Segment'][normalised_data['Condition']=='AIR'].unique())!=3):
            print('!!!Air segment missing')
        if (len(normalised_data['Segment'][normalised_data['Condition']=='CO2'].unique())!=3):
            print('!!!CO2 segment missing')
        return normalised_data
    
    def normalise_data(participant_df):
        data_to_normalise = participant_df.copy()
        #Remove data before calibration and after condition ended
        data_to_normalise = data_to_normalise[data_to_normalise['Segment'].notna()].reset_index(drop=True)
        normalised_data = data_to_normalise.copy()
        
        for col_name, col_data in data_to_normalise.iteritems():
            #if column needs to be normalised
            if(col_name in constants.NORMALISATION_COLUMNS):
                normalised_data[col_name] = min_max_normalisation(col_data)
                
                #normalised_data[col_name][air_data.index[0]:air_data.index[len(normalised_air_data)-1]+1] = normalised_air_data.reshape(-1)
                #normalised_data[col_name][co2_data.index[0]:co2_data.index[len(normalised_co2_data)-1]+1] = normalised_co2_data.reshape(-1)
                
        #Check that all segments and condition labels are prescent and that GSR and Eye tracking have been synced
        if (len(normalised_data['Condition'].unique())!=2):
            print('!!!Condition missing')
        if (len(normalised_data['Segment'][normalised_data['Condition']=='AIR'].unique())!=3):
            print('!!!Air segment missing')
        if (len(normalised_data['Segment'][normalised_data['Condition']=='CO2'].unique())!=3):
            print('!!!CO2 segment missing')
        return normalised_data
    
    def label_data(data):
        data.insert(2, 'Segment', np.nan)
        # Get start and end unix timestamps for expression calibration
        calibration_data_row_pair = data[data['Event']=='Calibration']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for expression calibration')
        else:
            data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1],'Segment'] = 'expression_calibration'
        
        # Get start and end unix timestamps for brightness calibration
        calibration_data_row_pair = data[data['Event']=='Brightness Calibration']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for brightness calibration')
        else:
            data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1],'Segment'] = 'brightness_calibration'
        
        # Get start and end unix timestamps for expression calibration
        calibration_data_row_pair = data[data['Event']=='Condition 1']
        if(len(calibration_data_row_pair)!=2):
            print('Invalid events for gas inhalation')
        else:
            data.loc[calibration_data_row_pair.index[0]:calibration_data_row_pair.index[1],'Segment'] = 'gas_inhalation'
        return data
    
    def filter_data(participant_df):
        filtered_data = participant_df.copy()
        filtered_data = filters.filter_pupil_size(filtered_data)
        filtered_data = filters.filter_biopac_gsr(filtered_data)
        filtered_data = filters.filter_biopac_respiration(filtered_data)
        filtered_data = filters.filter_ppg(filtered_data)
        filtered_data = filters.filter_contact(filtered_data)
        filtered_data = filters.filter_acc(filtered_data)
        filtered_data = filters.filter_gyr(filtered_data)
        
        return filtered_data
    
    def extract_features(participant_df):
        
        features_directory = os.path.join(os.getcwd(), 'temp', 'features')
        participant_number = participant_df['Participant_No'].unique()
        
        if(len(participant_number)!=1):
            print('Invalid participant file')
            return np.nan
        else:
            participant_number = participant_number[0]
        
        if (not os.path.exists(features_directory)):
            os.mkdir(features_directory)
        participant_features_file = os.path.join(features_directory, participant_number + '.csv')
        if (not os.path.exists(participant_features_file)):
        
            print('Extracting features for participant: ' + str(participant_number))
            df = participant_df.copy()
            df = df[~df['Segment'].isna()].reset_index(drop=True)
    
            #columns_to_calculate = ['Biopac_GSR', 'Biopac_RSP']  # Specify the columns to calculate features for
            columns_to_calculate = constants.FEATURE_EXTRACTION_COLUMNS
            result = pd.DataFrame()
            
            for condition in df['Condition'].unique():
                condition_df = df[df['Condition']==condition]
                for segment in df['Segment'].unique():
                    segment_df = condition_df[condition_df['Segment']==segment]
    
                    windows = generate_sliding_windows(segment_df, 5, 3)
                    segment_result = pd.DataFrame()
                    for window in windows:
                        window_features = pd.DataFrame()
                        for column_name, column_data in window[columns_to_calculate].iteritems(): 
                            features = calculate_statistical_features(column_data, column_name)
                            if(window_features.empty):
                                window_features = pd.DataFrame([features])
                            else:
                                window_features = pd.concat([window_features, pd.DataFrame([features])], axis=1)
                        segment_result = pd.concat([segment_result, window_features], ignore_index=True)
                    segment_result.insert(0, 'Condition', condition)
                    segment_result.insert(1, 'Segment', segment)
                    result = pd.concat([result, segment_result]).reset_index(drop=True)
            
            result.to_csv(participant_features_file)
            return result
        else:
            print('Features file found. Loading features file for participant: ' + str(participant_number))
            return  pd.read_csv(participant_features_file, index_col=0)
    
    



    




  