# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 02:07:08 2023

@author: m
"""
import pandas as pd
import numpy as np

def averge_fill(data):
    #iterates over dataframe column to look for -1 values prescent in eye tracking data. If -1 value is found, it is replaced by a mean value of the
    #previous and next non -1 values
    # iterate over each row
    df = pd.DataFrame(data.copy())

    for i in range(df.shape[0]):
        # if the current row contains -1 value
        if -1 in df.iloc[i].values:
            # get the indices of first non -1 frames before and after the current row
            prev_idx = max(np.where(df.iloc[:i, :] != -1)[0], default=-1)
            next_idx = min(np.where(df.iloc[i+1:, :] != -1)[0], default=df.shape[0]-1) + i + 1
            if(next_idx<len(df)):  
                # calculate the mean of the values in the time window
                mean_val = float((df.loc[df.index[prev_idx]] + df.loc[df.index[next_idx]])/2)
                # replace the -1 values in the current row with the calculated mean
                if(mean_val<1.5):
                    print('ERROR')
                df.iloc[i, :] = np.where(df.iloc[i, :] == -1, mean_val, df.iloc[i, :])
            else:
                df.iloc[i, :] = df.loc[prev_idx]
    return df

def impute_eye_data_old(eye_df):

    data = pd.read_csv(eye_df)
    if(not data.empty):
        data = data.drop(index=range(10)).reset_index(drop=True)
        
        # Mark pupil sizes outside pre-defined range as invalid for imputing
        conditions = ((data['VerboseData.Right.PupilDiameterMm'] < 1.5) |
                      (data['VerboseData.Right.PupilDiameterMm'] > 9) |
                      (data['VerboseData.Left.PupilDiameterMm'] < 1.5) |
                      (data['VerboseData.Left.PupilDiameterMm'] > 9))
        # Set the values to -1 based on the conditions
        data.loc[conditions, ['VerboseData.Right.PupilDiameterMm', 'VerboseData.Left.PupilDiameterMm']] = -1
        
        print('Imputing eye tracking data')
        right_pupil = averge_fill(data['VerboseData.Right.PupilDiameterMm'])
        left_pupil = averge_fill(data['VerboseData.Left.PupilDiameterMm'])
        data = pd.concat([data['TimestampUnix'], data['TimestampJ2000'], right_pupil, left_pupil], axis = 1)
    else:
        print('Empty eye data file')
    return data

def impute_eye_data(eye_df):

    data = pd.read_csv(eye_df)
    if(not data.empty):
        data = data.drop(index=range(10)).reset_index(drop=True)
        
        # Mark pupil sizes outside pre-defined range as invalid for imputing
        conditions = ((data['VerboseData.Right.PupilDiameterMm'] < 1.5) |
                      (data['VerboseData.Right.PupilDiameterMm'] > 9) |
                      (data['VerboseData.Left.PupilDiameterMm'] < 1.5) |
                      (data['VerboseData.Left.PupilDiameterMm'] > 9))
        # Set the values to -1 based on the conditions
        data.loc[conditions, ['VerboseData.Right.PupilDiameterMm', 'VerboseData.Left.PupilDiameterMm']] = -1
        
        # Calculate the absolute change between samples for the left pupil
        absolute_change_left = np.abs(data['VerboseData.Left.PupilDiameterMm'].diff())
        # Calculate the temporal separation between samples for the left pupil
        temporal_separation_left = data['TimestampUnix'].diff()
        # Calculate the normalized dilation speed for the left pupil
        dilation_speed_left = absolute_change_left / temporal_separation_left
        # Calculate the dilation speed at each sample for the left pupil
        d_prime_left = np.maximum(dilation_speed_left.shift(1), dilation_speed_left.shift(-1))
        # Calculate the median absolute deviation (MAD) for the left pupil
        mad_left = np.nanmedian(np.abs(d_prime_left - np.nanmedian(d_prime_left)))
        # Define the constant multiplier 'n' for the left pupil
        n_left = 0  # Adjust this value based on your specific requirements
        # Calculate the threshold for the left pupil
        threshold_left = np.nanmedian(d_prime_left) + n_left * mad_left
        # Mark samples with dilation speeds above the threshold as outliers for the left pupil
        outliers_left = data[d_prime_left > threshold_left]
        # Set the values of the left pupil to -1 for the outliers
        data.loc[d_prime_left > threshold_left, 'VerboseData.Left.PupilDiameterMm'] = -1
        # Repeat the same calculations for the right pupil
        absolute_change_right = np.abs(data['VerboseData.Right.PupilDiameterMm'].diff())
        temporal_separation_right = data['TimestampUnix'].diff()
        dilation_speed_right = absolute_change_right / temporal_separation_right
        d_prime_right = np.maximum(dilation_speed_right.shift(1), dilation_speed_right.shift(-1))
        mad_right = np.nanmedian(np.abs(d_prime_right - np.nanmedian(d_prime_right)))
        n_right = 0  # Adjust this value based on your specific requirements
        threshold_right = np.nanmedian(d_prime_right) + n_right * mad_right
        outliers_right = data[d_prime_right > threshold_right]
        data.loc[d_prime_right > threshold_right, 'VerboseData.Right.PupilDiameterMm'] = -1
        
        adjacent_rows_to_discard = 1
        mask = data['VerboseData.Left.PupilDiameterMm'] == -1
        
        # Get the indices of the rows where the condition is True
        indices = data.index[mask]
        
        # Set the adjacent rows to -1
        valid_indices = indices[(indices - adjacent_rows_to_discard >= 0) & (indices + adjacent_rows_to_discard < len(data))]
        data.loc[valid_indices - adjacent_rows_to_discard, 'VerboseData.Left.PupilDiameterMm'] = -1
        data.loc[valid_indices + adjacent_rows_to_discard, 'VerboseData.Left.PupilDiameterMm'] = -1
        
        mask = data['VerboseData.Right.PupilDiameterMm'] == -1
        
        # Get the indices of the rows where the condition is True
        indices = data.index[mask]
        
        # Set the adjacent rows to -1
        valid_indices = indices[(indices - adjacent_rows_to_discard >= 0) & (indices + adjacent_rows_to_discard < len(data))]
        data.loc[valid_indices - adjacent_rows_to_discard, 'VerboseData.Right.PupilDiameterMm'] = -1
        data.loc[valid_indices + adjacent_rows_to_discard, 'VerboseData.Right.PupilDiameterMm'] = -1


        # Replace -1 with NaN to represent missing values
        data = data.replace(-1, np.nan)
        
        # Calculate the mean of the nearest non -1 values for the left pupil column
        left_pupil_mean = data['VerboseData.Left.PupilDiameterMm'].interpolate(method='linear')
        
        # Update the left pupil column with the calculated mean values
        data['VerboseData.Left.PupilDiameterMm'] = left_pupil_mean
        
        # Calculate the mean of the nearest non -1 values for the right pupil column
        right_pupil_mean = data['VerboseData.Right.PupilDiameterMm'].interpolate(method='linear')
        
        # Update the right pupil column with the calculated mean values
        data['VerboseData.Right.PupilDiameterMm'] = right_pupil_mean

        print('Imputing eye tracking data')
        #right_pupil = averge_fill(data['VerboseData.Right.PupilDiameterMm'])
        #left_pupil = averge_fill(data['VerboseData.Left.PupilDiameterMm'])
        right_pupil = data['VerboseData.Right.PupilDiameterMm']
        left_pupil = data['VerboseData.Left.PupilDiameterMm']
        data = pd.concat([data['TimestampUnix'], data['TimestampJ2000'], right_pupil, left_pupil], axis = 1)
    else:
        print('Empty eye data file')
    return data