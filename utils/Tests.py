# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:06:56 2023

@author: m
"""

from scipy import stats
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

class Tests:

    def normality_test_sw(condition_data, condition_name):
        # Shapiro-Wilk test for normality
        shapiro_test = stats.shapiro(condition_data)
        print(condition_name)
        print("Shapiro-Wilk Test - Statistic:", shapiro_test.statistic)
        print("Shapiro-Wilk Test - p-value:", shapiro_test.pvalue)
        if (shapiro_test.pvalue>0.05):
            normality_string = ""
        else:
            normality_string = "NOT"
        print("Data for " + condition_name + " condition is " + normality_string + " normally distributed.")
    
        return shapiro_test
        
    def paired_t_test(condition_1_data, condition_2_data):
        # Perform the paired t-test
        t_statistic, p_value = stats.ttest_rel(condition_1_data, condition_2_data, nan_policy='omit')
        print(f"T-statistic: {t_statistic:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        if (p_value>0.05):
            results_string = "No singificant results found."
        else:
            results_string = "SIGNIFICANT difference found."
        print(results_string)
        
        return t_statistic, p_value
    
    def regression_tests_linear(air_windows, co2_windows, combined_windows, dependent_variable):
        print('Running linear regression model independently for air and co2 conditions for: ' + dependent_variable)
        # fit a linear model for the Air condition
        air_model = smf.ols(formula=(dependent_variable + ' ~ window_index'), data=air_windows).fit()
        #print('AIR MODEL')
        #print(air_model.summary())
        
        # fit a linear model for the CO2 condition
        co2_model = smf.ols(formula=(dependent_variable + ' ~ window_index'), data=co2_windows).fit()
        #print('CO2 MODEL')
        #print(co2_model.summary())
        
        # compare the slopes of the two models using an F-test
        f_statistic, p_value, _ = air_model.compare_f_test(co2_model)
        print(f"F-statistic: {f_statistic:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        # extract coefficients for air condition
        air_coef = air_model.params
        air_window_coef = air_coef['window_index']
        air_intercept_coef = air_coef['Intercept']
        
        # extract coefficients for CO2 condition
        co2_coef = co2_model.params
        co2_window_coef = co2_coef['window_index']
        co2_intercept_coef = co2_coef['Intercept']
        
        print(f"Air condition: {dependent_variable} increases by {air_window_coef:.3f} units per window")
        print(f"CO2 condition: {dependent_variable} increases by {co2_window_coef:.3f} units per window")
        #print(f"{dependent_variable} at the start of the AIR condition is on average: {air_intercept_coef:.3f}")
        #print(f"{dependent_variable} at the start of the CO2 condition is on average: {co2_intercept_coef:.3f}")
        
        print('Running one linear regression model with condition as an independent variable: ' + dependent_variable)
        
        # Set up the linear regression model
        model = smf.ols(dependent_variable + ' ~ Condition + window_index', data=combined_windows)
        
        # Fit the model
        results = model.fit()

        condition_coef = results.params['Condition[T.CO2]']
        
        print(f"{dependent_variable} in CO2 condition is on average higher by {condition_coef:.3f}")

        # Print the model summary
        print(results.summary())

        
    def regression_tests_mixed_linear(air_windows, co2_windows, combined_windows, dependent_variable):
        print('Running mixed linear regression model independently for air and co2 conditions for: ' + dependent_variable)
        # fit a linear model for the Air condition
        air_model = smf.mixedlm(formula=(dependent_variable + ' ~ window_index'), data=air_windows, groups=air_windows["participant_number"]).fit()
        #print('AIR MODEL')
        #print(air_model.summary())
        
        # fit a linear model for the CO2 condition
        co2_model = smf.mixedlm(formula=(dependent_variable + ' ~ window_index'), data=co2_windows, groups=co2_windows["participant_number"]).fit()
        #print('CO2 MODEL')
        #print(co2_model.summary())
        
        # extract coefficients for air condition
        air_coef = air_model.params
        air_window_coef = air_coef['window_index']
        air_intercept_coef = air_coef['Intercept']
        
        # extract coefficients for CO2 condition
        co2_coef = co2_model.params
        co2_window_coef = co2_coef['window_index']
        co2_intercept_coef = co2_coef['Intercept']
        
        print(f"Air condition: {dependent_variable} increases by {air_window_coef:.3f} units per window")
        print(f"CO2 condition: {dependent_variable} increases by {co2_window_coef:.3f} units per window")
        #print(f"{dependent_variable} at the start of the AIR condition is on average: {air_intercept_coef:.3f}")
        #print(f"{dependent_variable} at the start of the CO2 condition is on average: {co2_intercept_coef:.3f}")
        
        print('Running one mixed linear regression model with condition as an independent variable: ' + dependent_variable)
        
        # Set up the linear regression model
        model = smf.mixedlm(dependent_variable + ' ~ window_index * Condition', data=combined_windows, groups=combined_windows["participant_number"])
        # Fit the model
        results = model.fit()

        condition_coef = results.params['Condition[T.CO2]']
        
        print(f"{dependent_variable} in CO2 condition is on average higher by {condition_coef:.3f}")

        # Print the model summary
        print(results.summary())
        
    def regression_tests_polynomial(air_windows, co2_windows, combined_windows, dependent_variable, power):
        print('Running mixed polynomial regression model independently for air and co2 conditions for: ' + dependent_variable)
        
        ## AIR_________________
        air_model = smf.mixedlm(formula=(dependent_variable + ' ~ window_index + np.power(window_index, '+ str(power) +')'), data=air_windows, groups=air_windows["participant_number"]).fit()
        print('AIR MODEL')
        print(air_model.summary())
        
        # Generate a range of x values for plotting
        air_x_values = np.linspace(air_windows['window_index'].min(), air_windows['window_index'].max(), 100)
        
        # Calculate the predicted values using the fitted model
        air_predicted_values = air_model.predict({'window_index': air_x_values})
        
        ## CO2_________________
        co2_model = smf.mixedlm(formula=(dependent_variable + ' ~ window_index + np.power(window_index, '+ str(power) +')'), data=co2_windows, groups=co2_windows["participant_number"]).fit()
        print('CO2 MODEL')
        print(co2_model.summary())
        
        # Generate a range of x values for plotting
        co2_x_values = np.linspace(air_windows['window_index'].min(), co2_windows['window_index'].max(), co2_windows['window_index'].max()+1)
        
        # Calculate the predicted values using the fitted model
        co2_predicted_values = co2_model.predict({'window_index': co2_x_values})
        
        # extract coefficients for air condition
        air_coef = air_model.params
        air_window_coef = air_coef['window_index']
        air_intercept_coef = air_coef['Intercept']
        
        # extract coefficients for CO2 condition
        co2_coef = co2_model.params
        co2_window_coef = co2_coef['window_index']
        co2_intercept_coef = co2_coef['Intercept']
        
        print(f"Air condition: {dependent_variable} increases by {air_window_coef:.3f} units per window")
        print(f"CO2 condition: {dependent_variable} increases by {co2_window_coef:.3f} units per window")
        #print(f"{dependent_variable} at the start of the AIR condition is on average: {air_intercept_coef:.3f}")
        #print(f"{dependent_variable} at the start of the CO2 condition is on average: {co2_intercept_coef:.3f}")
        
        print('Running one mixed polynomial regression model with condition as an independent variable: ' + dependent_variable)
        
        model = smf.mixedlm(dependent_variable + ' ~ window_index + Condition', data=combined_windows, groups=combined_windows["participant_number"])
        # Fit the model
        results = model.fit()

        condition_coef = results.params['Condition[T.CO2]']
        
        print(f"{dependent_variable} in CO2 condition is on average higher by {condition_coef:.3f}")

        # # Print the model summary
        print('COMBINED MODEL')
        print(results.summary())
        
        
        return [air_x_values, air_predicted_values], [co2_x_values, co2_predicted_values]
    
    def correlation_test(data_windows):
        
        # Generate correlation matrix
        # create a list of column names that contain "mean" or "Mean"
        mean_columns = data_windows.columns[data_windows.columns.str.contains('_mean|_Mean|HRV|SCR_Peaks_N|RSP_Phase_Duration_Ratio|EDA_Tonic_SD|pupil_size_combined')].tolist()
        mean_columns = list(filter(lambda item: 'derivative' not in item, mean_columns))
        mean_columns = list(filter(lambda item: 'Filtered' not in item, mean_columns))
        mean_columns.remove('Ppg/Raw.ppg_mean')
        
        # select only the columns that contain "mean" or "Mean"
        df = data_windows[mean_columns]
        #df_mean = df_mean.dropna()
        
        # Shorten column names
        df.rename(columns={
            'Emg/Contact[RightOrbicularis]_mean': 'Emg/C(RO)',
            'Emg/Amplitude[RightOrbicularis]_mean': 'Emg/A(RO)',
            'Emg/Contact[RightZygomaticus]_mean': 'Emg/C(RZ)',
            'Emg/Amplitude[RightZygomaticus]_mean': 'Emg/A(RZ)',
            'Emg/Contact[CenterCorrugator]_mean': 'Emg/C(CC)',
            'Emg/Amplitude[CenterCorrugator]_mean': 'Emg/A(CC)',
            'Emg/Contact[RightFrontalis]_mean': 'Emg/C(RF)',
            'Emg/Amplitude[RightFrontalis]_mean': 'Emg/A(RF)',
            'Emg/Contact[LeftOrbicularis]_mean': 'Emg/C(LO)',
            'Emg/Amplitude[LeftOrbicularis]_mean': 'Emg/A(LO)',
            'Emg/Contact[LeftZygomaticus]_mean': 'Emg/C(LZ)',
            'Emg/Amplitude[LeftZygomaticus]_mean': 'Emg/A(LZ)',
            'Emg/Contact[LeftFrontalis]_mean': 'Emg/C(LF)',
            'Emg/Amplitude[LeftFrontalis]_mean': 'Emg/A(LF)',
            'Emg/Amplitude[LeftFrontalis]_mean': 'Emg/A(LF)',
            
            'HeartRate/Average_mean': 'HeartRate(EmteqPro)',
            'PPG_Rate_Mean': 'HeartRate(feature)',
            'Accelerometer/Raw.x_mean': 'Acc(X)',
            'Accelerometer/Raw.y_mean': 'Acc(Y)',
            'Accelerometer/Raw.z_mean': 'Acc(Z)',
            'Gyroscope/Raw.x_mean': 'Gyr(X)',
            'Gyroscope/Raw.y_mean': 'Gyr(Y)',
            'Gyroscope/Raw.z_mean': 'Gyr(Z)',
            'pupil_size_combined': 'PupilSize(Combined)',
            'VerboseData.Right.PupilDiameterMm_mean': 'PupilSize(Right)',
            'VerboseData.Left.PupilDiameterMm_mean': 'PupilSize(Left)',
            'Biopac_GSR_mean': 'GSR',
            'SCR_Peaks_Amplitude_Mean': 'SCR_Peaks_Amp',
            'Biopac_RSP_mean': 'RSP',
            'RSP_Rate_Mean': 'RSP_Rate',
            'RSP_Amplitude_Mean': 'RSP_Amp',
            'RSP_Phase_Duration_Ratio': 'RSP_Phase_Dur_Ratio',
            
            # Add the remaining column name mappings here
        }, inplace=True)
        

        # compute the correlation matrix
        corr_matrix = df.corr(method=lambda x, y: pearsonr(x, y)[0])
        
        # compute the p-values for each correlation coefficient
        p_values = df.corr(method=lambda x, y: pearsonr(x, y)[1])
        
        # adjust the p-values using FDR control
        reject, p_values_fdr = multipletests(p_values.values.flatten(), alpha=0.05, method='fdr_by')[:2]
        p_values_fdr = pd.DataFrame(p_values_fdr.reshape(p_values.shape), index=p_values.index, columns=p_values.columns)
        
        # use the adjusted p-values to filter the correlation matrix
        #corr_matrix_fdr = corr_matrix.where(p_values_fdr < 0.05)
        #p_values_fdr = p_values_fdr.where(p_values_fdr < 0.05)
        
        
        # dont filter based on fdr
        corr_matrix_fdr = corr_matrix
        
        return corr_matrix_fdr
        
        # Save CSV
        #corr_matrix_fdr.to_csv(os.path.join(notebook_temp_dir, "correlation_matrix_test_statistic.csv"))
        #p_values_fdr.to_csv(os.path.join(notebook_temp_dir, "correlation_matrix_p_values.csv"))




    

        



    



        

