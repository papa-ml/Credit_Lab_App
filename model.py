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
import pickle
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Import Functions
from functions import preprocess_inputs_pd, preprocess_inputs_lgd, select_model_variables, LogisticRegression_with_p_values, LinearRegression, adjust_scorecard

# Import Variables
from model_training_variables import url, reference_date, pd_variable_list, pd_ref_categories, lgd_features_all, lgd_features_reference_cat, min_score, max_score




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

'''# Producing the sample credit scores
#####Entire section might need to be placed on app file#####
#***! This is used for producing the example credit scores'''
inputs_test_with_ref_cat = select_model_variables(loan_data_inputs_test, pd_variable_list) #This will the input file from the user
inputs_test = inputs_test_with_ref_cat.drop(pd_ref_categories, axis = 1)

inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat
inputs_test_with_ref_cat_w_intercept.insert(0,'Intercept',1)
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
scorecard_scores = df_scorecard['Score - Final']
scorecard_scores = scorecard_scores.values.reshape(102,1)
y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
'''#y_scores is the output of the predicted credit score
#print(y_scores)'''
sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
y_hat_proba_from_score = np.exp(sum_coef_from_score) / (1 + np.exp(sum_coef_from_score))
'''y_hat_proba_from_score is the output for PD
print(y_hat_proba_from_score)'''
#### The Credit Score Section - Not Needed for Model, but can be displayed

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

feature_name = lgd_inputs_stage_1_train.columns.values

#LGD Summary Table from Stage 1
#***Need to update this to a function since used in other sections
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg_lgd_st_1.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_lgd_st_1.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg_lgd_st_1.p_values
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values

#****End of update section

# Stage 2
lgd_stage_2_data = loan_data_defaults[loan_data_defaults['recovery_rate_0_1'] == 1]
lgd_inputs_stage_2_train, lgd_inputs_stage_2_test, lgd_targets_stage_2_train, lgd_targets_stage_2_test = train_test_split(lgd_stage_2_data.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), lgd_stage_2_data['recovery_rate'], test_size = 0.2, random_state = 42)
lgd_inputs_stage_2_train = lgd_inputs_stage_2_train[lgd_features_all]
lgd_inputs_stage_2_train = lgd_inputs_stage_2_train.drop(lgd_features_reference_cat, axis = 1)
reg_lgd_st_2 = LinearRegression()
reg_lgd_st_2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)

feature_name = lgd_inputs_stage_2_train.columns.values

#LGD Summary Table from Stage 2
#***Need to update this to a function since used in other sections
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg_lgd_st_2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_lgd_st_2.intercept_]
summary_table = summary_table.sort_index()
p_values = reg_lgd_st_2.p
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values.round(3)
#print(summary_table)

#****End of update section

#!!! Producing EAD Model
ead_inputs_train, ead_inputs_test, ead_targets_train, ead_targets_test = train_test_split(loan_data_defaults.drop(['good_bad', 'recovery_rate','recovery_rate_0_1', 'CCF'], axis = 1), loan_data_defaults['CCF'], test_size = 0.2, random_state = 42)
ead_inputs_train = ead_inputs_train[lgd_features_all]
ead_inputs_train = ead_inputs_train.drop(lgd_features_reference_cat, axis = 1)
reg_ead = LinearRegression()
reg_ead.fit(ead_inputs_train, ead_targets_train)
feature_name = ead_inputs_train.columns.values

#EAD Summary Table
#***Need to update this to a function since used in other sections
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
summary_table['Coefficients'] = np.transpose(reg_ead.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_ead.intercept_]
summary_table = summary_table.sort_index()
p_values = reg_ead.p
p_values = np.append(np.nan,np.array(p_values))
summary_table['p_values'] = p_values
#print(summary_table)
#****End of update section

# This section needs cleaned up and reorganized

loan_data_preprocessed_lgd_ead = loan_data_lgd[lgd_features_all]
loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(lgd_features_reference_cat, axis = 1)

#These predictions take the model from this page and input data from the user
#Predict LGD Stage 1 and 2
loan_data_lgd['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(loan_data_preprocessed_lgd_ead)
loan_data_lgd['recovery_rate_st_2'] = reg_lgd_st_2.predict(loan_data_preprocessed_lgd_ead)
loan_data_lgd['recovery_rate'] = loan_data_lgd['recovery_rate_st_1'] * loan_data_lgd['recovery_rate_st_2']
loan_data_lgd['recovery_rate'] = np.where(loan_data_lgd['recovery_rate'] < 0, 0, loan_data_lgd['recovery_rate'])
loan_data_lgd['recovery_rate'] = np.where(loan_data_lgd['recovery_rate'] > 1, 1, loan_data_lgd['recovery_rate'])
loan_data_lgd['LGD'] = 1 - loan_data_lgd['recovery_rate']

# Predict EAD
loan_data_lgd['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)
loan_data_lgd['CCF'] = np.where(loan_data_lgd['CCF'] < 0, 0, loan_data_lgd['CCF'])
loan_data_lgd['CCF'] = np.where(loan_data_lgd['CCF'] > 1, 1, loan_data_lgd['CCF'])
loan_data_lgd['EAD'] = loan_data_lgd['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

# Predict PD
loan_data_inputs_pd = pd.concat([loan_data_inputs_train, loan_data_inputs_test], axis = 0)
loan_data_inputs_pd_temp = loan_data_inputs_pd[pd_variable_list]
loan_data_inputs_pd_temp = loan_data_inputs_pd_temp.drop(pd_ref_categories, axis = 1)
loan_data_inputs_pd['PD'] = reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[: ][: , 0]
loan_data_preprocessed_new = pd.concat([loan_data_lgd, loan_data_inputs_pd], axis = 1)



# End Result
loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['PD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['EAD']

#Expected Loss Components and Expected Loss
#Export this CSV
loan_data_preprocessed_new[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head()

#Portfolio Expected Loss
# Sum of Expected Loss
loan_data_preprocessed_new['EL'].sum()

#Portfolio Notional Value
#Sum of Funded or Face Amount
loan_data_preprocessed_new['funded_amnt'].sum()

print(loan_data_preprocessed_new['EL'].sum() / loan_data_preprocessed_new['funded_amnt'].sum())
print(loan_data_preprocessed_new[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head())






