import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from functions import normalize_date_string, woe_discrete
from sklearn.model_selection import train_test_split

# Data Import
csv_url = "https://drive.google.com/uc?id=1sOwqCmj5Qn0_fexszv7Pk0VPfqZQN1_5"
reference_date = '2017-12-01'

loan_data_backup = pd.read_csv(csv_url)
loan_data = loan_data_backup.copy()

# Cleaning
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace(r'\+ years', '', regex=True)
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

loan_data['earliest_cr_line_clean'] = loan_data['earliest_cr_line'].apply(normalize_date_string)
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line_clean'],format='%b-%y',errors='coerce')
loan_data['mths_since_earliest_cr_line'] = ((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']).dt.days / 30.44).round()
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()
loan_data['mths_since_earliest_cr_line'] = ((pd.to_datetime('2017-12-01') - loan_data['mths_since_issue_d']).dt.days / 30.44).round()

loan_data['issue_d_padded'] = loan_data['issue_d'].str.replace(r'^(\d{1})-', r'0\1-', regex=True)
loan_data['mths_since_issue_d'] = pd.to_datetime(loan_data['issue_d_padded'],format='%y-%b')

loan_data['term_int'] = loan_data['term'].str.replace(' months', '')
loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
columns_to_fill = ['mths_since_earliest_cr_line', 'acc_now_delinq', 'total_acc','pub_rec','open_acc','inq_last_6mths','delinq_2yrs','emp_length_int']
loan_data[columns_to_fill] = loan_data[columns_to_fill].fillna(0)

loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']), 0, 1)

#Dummy Variables
pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep= ":").astype(int)
loan_data_dummies = [
    pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep= ":").astype(int),
    pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep= ":").astype(int),
]
loan_data_dummies = pd.concat(loan_data_dummies, axis=1)
loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)


loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size=0.2, random_state=42)
df_inputs_prepr = loan_data_inputs_train
df_targets_prepr = loan_data_targets_train

df_temp = woe_discrete(df_inputs_prepr, 'grade', df_targets_prepr)
df_temp