# Import Libraries
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat
from joblib import dump
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import os

# Import Functions
from functions import preprocess_inputs_pd, preprocess_inputs_lgd, select_model_variables, LogisticRegression_with_p_values, LinearRegression, adjust_scorecard

# Import Variables
from model_training_variables import url, reference_date, pd_variable_list, pd_ref_categories, lgd_features_all, lgd_features_reference_cat, min_score, max_score

#Code to Initiate Training
def train_models_if_needed():
    if os.path.exists("pd_model.joblib") and \
       os.path.exists("lgd_stage_1_model.joblib") and \
       os.path.exists("lgd_stage_2_model.joblib") and \
       os.path.exists("ead_model.joblib"):
        return

    # Data Import
    loan_data_backup = pd.read_csv(url)
    loan_data = loan_data_backup.copy()

    # Data Cleaning
    loan_data_pd = preprocess_inputs_pd(loan_data, reference_date)
    loan_data_lgd = preprocess_inputs_lgd(loan_data, reference_date)

    # Train Test Split
    loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data_pd.drop('good_bad', axis = 1), loan_data_pd['good_bad'], test_size = 0.2, random_state = 42)

    # Select Dummy Variables and Remove Reference Categories
    inputs_train_with_ref_cat = select_model_variables(loan_data_inputs_train, pd_variable_list)
    inputs_train = inputs_train_with_ref_cat.drop(pd_ref_categories, axis = 1)

    #PD Model Fitting
    reg_pd = LogisticRegression_with_p_values()
    reg_pd.fit(inputs_train, loan_data_targets_train)
    dump(reg_pd, "pd_model.joblib")

    #PD Summary Table
    #***Need to update this to a function since used in other sections
    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg_pd.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg_pd.intercept_[0]]
    summary_table = summary_table.sort_index()
    p_values = reg_pd.p_values
    p_values = np.append(np.nan,np.array(p_values))
    summary_table['p_values'] = p_values
    #****End of update section

    #Creates Scorecard
    df_ref_categories = pd.DataFrame(pd_ref_categories, columns = ['Feature name'])
    df_ref_categories['Coefficients'] = 0
    df_ref_categories['p_values'] = np.nan
    df_scorecard = pd.concat([summary_table, df_ref_categories])
    df_scorecard = df_scorecard.reset_index()
    df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
    min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
    max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
    df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
    df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
    df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
    df_scorecard = adjust_scorecard(df_scorecard)


    # !!!!Producing LGD Model
    # Stage 1
    loan_data_defaults = loan_data_lgd[loan_data_lgd['loan_status'].isin(['Charged Off','Does not meet the credit policy. Status:Charged Off'])]
    loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])
    loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']
    loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)
    lgd_inputs_stage_1_train, lgd_inputs_stage_1_test, lgd_targets_stage_1_train, lgd_targets_stage_1_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['recovery_rate_0_1'], test_size = 0.2, random_state = 42)
    lgd_inputs_stage_1_train = lgd_inputs_stage_1_train[lgd_features_all]
    lgd_inputs_stage_1_train = lgd_inputs_stage_1_train.drop(lgd_features_reference_cat, axis = 1)

    reg_lgd_st_1 = LogisticRegression_with_p_values(solver='liblinear', max_iter=1000)
    reg_lgd_st_1.fit(lgd_inputs_stage_1_train, lgd_targets_stage_1_train)
    dump(reg_lgd_st_1, "lgd_stage_1_model.joblib")

    # Stage 2
    lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
    lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)
    lgd_inputs_stage_2_train = lgd_inputs_stage_2_train[lgd_features_all]
    lgd_inputs_stage_2_train = lgd_inputs_stage_2_train.drop(lgd_features_reference_cat, axis = 1)
    reg_lgd_st_2 = LinearRegression()
    reg_lgd_st_2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)
    dump(reg_lgd_st_2, "lgd_stage_2_model.joblib")

    #!!! Producing EAD Model
    ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
    ead_inputs_train = ead_inputs_train[lgd_features_all]
    ead_inputs_train = ead_inputs_train.drop(lgd_features_reference_cat, axis = 1)
    reg_ead = LinearRegression()
    reg_ead.fit(ead_inputs_train, ead_targets_train)
    dump(reg_ead, "ead_model.joblib")
















