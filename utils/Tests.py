# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:06:56 2023

@author: m
"""

from scipy import stats
import statsmodels.formula.api as smf

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
    
    def regression_tests(air_windows, co2_windows, combined_windows, dependent_variable):
        print('Running regression model independently for air and co2 conditions for: ' + dependent_variable)
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
        
        print(f"Air condition: {dependent_variable} increases by {air_window_coef:.4f} units per window")
        print(f"CO2 condition: {dependent_variable} increases by {co2_window_coef:.4f} units per window")
        #print(f"{dependent_variable} at the start of the AIR condition is on average: {air_intercept_coef:.4f}")
        #print(f"{dependent_variable} at the start of the CO2 condition is on average: {co2_intercept_coef:.4f}")
        
        print('Running one regression model with condition as an independent variable: ' + dependent_variable)
        
        # Set up the linear regression model
        model = smf.ols(dependent_variable + ' ~ Condition + window_index', data=combined_windows)
        
        # Fit the model
        results = model.fit()

        condition_coef = results.params['Condition[T.CO2]']
        
        print(f"{dependent_variable} in CO2 condition is on average higher by {condition_coef:.4f}")

        # Print the model summary
        print(results.summary())

        



        

