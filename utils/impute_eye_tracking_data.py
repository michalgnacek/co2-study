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
                df.iloc[i, :] = np.where(df.iloc[i, :] == -1, mean_val, df.iloc[i, :])
            else:
                df.iloc[i, :] = df.loc[prev_idx]
    return df

def impute_eye_data(eye_df):

    data = pd.read_csv(eye_df)
    if(not data.empty):
        data = data.drop(index=range(10)).reset_index(drop=True)
        print('Imputing eye tracking data')
        right_pupil = averge_fill(data['VerboseData.Right.PupilDiameterMm'])
        left_pupil = averge_fill(data['VerboseData.Left.PupilDiameterMm'])
        data = pd.concat([data['TimestampUnix'], data['TimestampJ2000'], right_pupil, left_pupil], axis = 1)
    else:
        print('Empty eye data file')
    return data