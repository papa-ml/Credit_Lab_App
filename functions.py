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



def normalize_date_string(x):
    # Ensure it's a string
    x = str(x)

    # If it's already in 'Mon-YY' or 'Mon-YYYY' format, keep it
    match_standard = re.fullmatch(r'[A-Za-z]{3}-\d{2,4}', x)
    if match_standard:
        return x

    # Match patterns like '1-Nov', '3-Feb', '12-Mar'
    match_alt = re.match(r'(\d{1,2})[-/]?([A-Za-z]{3})', x)
    if match_alt:
        year = match_alt.group(1).zfill(2)  # Pad to 2 digits
        month = match_alt.group(2).capitalize()
        return f"{month}-{year}"

    return None  # For anything else

def preprocess_inputs_lgd(df, reference_date):
    df_copy = df.copy()

    df_copy['emp_length_int'] = (
    df_copy['emp_length']
    .replace({
      r'\+ years': '',
      '< 1 year': '0',
      'n/a': '0',
      ' years': '',
      ' year': ''
    }, regex=True)
    )
    df_copy['emp_length_int'] = pd.to_numeric(df_copy['emp_length_int'])


    df_copy['earliest_cr_line_clean'] = df_copy['earliest_cr_line'].apply(normalize_date_string)
    df_copy['earliest_cr_line_date'] = pd.to_datetime(df_copy['earliest_cr_line_clean'], format='%b-%y', errors='coerce')
    df_copy['mths_since_earliest_cr_line'] = ((pd.to_datetime(reference_date) - df_copy['earliest_cr_line_date']).dt.days / 30.44).round()
    max_val = df_copy['mths_since_earliest_cr_line'].max(skipna=True)
    df_copy.loc[df_copy['mths_since_earliest_cr_line'] < 0, 'mths_since_earliest_cr_line'] = max_val

    df_copy['term_int'] = (df_copy['term'].str.replace(' months', '', regex=False))
    df_copy['term_int'] = pd.to_numeric(df_copy['term_int'])

    df_copy['issue_d_padded'] = pd.to_datetime(df_copy['issue_d'].str.replace(r'^(\d{1})-', r'0\1-', regex=True),format='%y-%b',errors='coerce')
    df_copy['mths_since_issue_d'] = (((pd.to_datetime(reference_date) - df_copy['issue_d_padded']).dt.days / 30.44).round())

    df_copy['total_rev_hi_lim'] = df_copy['total_rev_hi_lim'].fillna(df_copy['funded_amnt'])
    df_copy['annual_inc'] = df_copy['annual_inc'].fillna(df_copy['annual_inc'].mean())

    fill_zero_cols = [
      'mths_since_earliest_cr_line',
      'acc_now_delinq',
      'total_acc',
      'pub_rec',
      'open_acc',
      'inq_last_6mths',
      'delinq_2yrs',
      'emp_length_int'
    ]
    df_copy[fill_zero_cols] = df_copy[fill_zero_cols].fillna(0).astype(int)

    categorical_cols = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'loan_status', 'purpose', 'addr_state', 'initial_list_status'
    ]
    dummies = pd.get_dummies(df_copy[categorical_cols], prefix=categorical_cols, prefix_sep=':').astype(int)
    df_copy = pd.concat([df_copy, dummies], axis=1)

    df_copy['mths_since_last_delinq'].fillna(0, inplace=True)
    df_copy['mths_since_last_record'].fillna(0, inplace=True)

    charged_off_statuses = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off',
    'Late (31-120 days)'
    ]
    df_copy['good_bad'] = np.where(df_copy['loan_status'].isin(charged_off_statuses), 0, 1)

   # df_copy = df_copy.apply(pd.to_numeric, errors='coerce')

    # Handle missing values and ensure proper data types
   # df_copy.fillna(0, inplace=True)

    return df_copy


# Clean and Preprocess - Produces data for pd
def preprocess_inputs_pd(df,reference_date):
    df_copy = df.copy()

    df_copy['emp_length_int'] = (
        df_copy['emp_length']
        .replace({
            r'\+ years': '',
            '< 1 year': '0',
            'n/a': '0',
            ' years': '',
            ' year': ''
        }, regex=True)
    )
    df_copy['emp_length_int'] = pd.to_numeric(df_copy['emp_length_int'])

    df_copy['earliest_cr_line_clean'] = df_copy['earliest_cr_line'].apply(normalize_date_string)
    df_copy['earliest_cr_line_date'] = pd.to_datetime(df_copy['earliest_cr_line_clean'], format='%b-%y', errors='coerce')
    df_copy['mths_since_earliest_cr_line'] = ((pd.to_datetime(reference_date) - df_copy['earliest_cr_line_date']).dt.days / 30.44).round()
    max_val = df_copy['mths_since_earliest_cr_line'].max(skipna=True)
    df_copy.loc[df_copy['mths_since_earliest_cr_line'] < 0, 'mths_since_earliest_cr_line'] = max_val

    df_copy['term_int'] = (df_copy['term'].str.replace(' months', '', regex=False).astype(int))

    df_copy['issue_d_padded'] = pd.to_datetime(df_copy['issue_d'].str.replace(r'^(\d{1})-', r'0\1-', regex=True),format='%y-%b',errors='coerce')
    df_copy['mths_since_issue_d'] = (((pd.to_datetime(reference_date) - df_copy['issue_d_padded']).dt.days / 30.44).round())

    df_copy['total_rev_hi_lim'] = df_copy['total_rev_hi_lim'].fillna(df_copy['funded_amnt'])
    df_copy['annual_inc'] = df_copy['annual_inc'].fillna(df_copy['annual_inc'].mean())

    fill_zero_cols = [
      'mths_since_earliest_cr_line',
      'acc_now_delinq',
      'total_acc',
      'pub_rec',
      'open_acc',
      'inq_last_6mths',
      'delinq_2yrs',
      'emp_length_int'
    ]
    df_copy[fill_zero_cols] = df_copy[fill_zero_cols].fillna(0).astype(int)

    categorical_cols = [
    'grade', 'sub_grade', 'home_ownership', 'verification_status',
    'loan_status', 'purpose', 'addr_state', 'initial_list_status'
    ]
    dummies = pd.get_dummies(df_copy[categorical_cols], prefix=categorical_cols, prefix_sep=':').astype(int)
    df_copy = pd.concat([df_copy, dummies], axis=1)

    charged_off_statuses = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off',
    'Late (31-120 days)'
    ]
    df_copy['good_bad'] = np.where(df_copy['loan_status'].isin(charged_off_statuses), 0, 1)

    rent_other_none_any_cols = [
        'home_ownership:RENT',
        'home_ownership:OTHER',
        'home_ownership:NONE',
        'home_ownership:ANY'
    ]
    for col in rent_other_none_any_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0
    new_columns = {
        'home_ownership:RENT_OTHER_NONE_ANY': df_copy[rent_other_none_any_cols].any(axis=1).astype(int)
    }
    new_cols_df = pd.DataFrame(new_columns, index=df_copy.index)
    df_copy = pd.concat([df_copy, new_cols_df], axis=1)

    if 'addr_state:ND' not in df_copy.columns:
        df_copy['addr_state:ND'] = 0

    state_groups = {
        'addr_state:ND_NE_IA_NV_FL_HI_AL': ['addr_state:ND', 'addr_state:NE', 'addr_state:IA', 'addr_state:NV',
                                            'addr_state:FL', 'addr_state:HI', 'addr_state:AL'],
        'addr_state:NM_VA': ['addr_state:NM', 'addr_state:VA'],
        'addr_state:OK_TN_MO_LA_MD_NC': ['addr_state:OK', 'addr_state:TN', 'addr_state:MO', 'addr_state:LA',
                                         'addr_state:MD', 'addr_state:NC'],
        'addr_state:UT_KY_AZ_NJ': ['addr_state:UT', 'addr_state:KY', 'addr_state:AZ', 'addr_state:NJ'],
        'addr_state:AR_MI_PA_OH_MN': ['addr_state:AR', 'addr_state:MI', 'addr_state:PA', 'addr_state:OH',
                                      'addr_state:MN'],
        'addr_state:RI_MA_DE_SD_IN': ['addr_state:RI', 'addr_state:MA', 'addr_state:DE', 'addr_state:SD',
                                      'addr_state:IN'],
        'addr_state:GA_WA_OR': ['addr_state:GA', 'addr_state:WA', 'addr_state:OR'],
        'addr_state:WI_MT': ['addr_state:WI', 'addr_state:MT'],
        'addr_state:IL_CT': ['addr_state:IL', 'addr_state:CT'],
        'addr_state:KS_SC_CO_VT_AK_MS': ['addr_state:KS', 'addr_state:SC', 'addr_state:CO', 'addr_state:VT',
                                         'addr_state:AK', 'addr_state:MS'],
        'addr_state:WV_NH_WY_DC_ME_ID': ['addr_state:WV', 'addr_state:NH', 'addr_state:WY', 'addr_state:DC',
                                         'addr_state:ME', 'addr_state:ID']
    }
    all_states = set(state for group in state_groups.values() for state in group)
    for state in all_states:
        if state not in df_copy.columns:
            df_copy[state] = 0
    new_columns = {
        group_col: df_copy[states].sum(axis=1).astype(int)
        for group_col, states in state_groups.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(new_columns, index=df_copy.index)], axis=1)

    purpose_groups = {
        'purpose:educ__sm_b__wedd__ren_en__mov__house': [
            'purpose:educational', 'purpose:small_business', 'purpose:wedding',
            'purpose:renewable_energy', 'purpose:moving', 'purpose:house'
        ],
        'purpose:oth__med__vacation': [
            'purpose:other', 'purpose:medical', 'purpose:vacation'
        ],
        'purpose:major_purch__car__home_impr': [
            'purpose:major_purchase', 'purpose:car', 'purpose:home_improvement'
        ]
    }
    all_purpose_cols = set(col for group in purpose_groups.values() for col in group)
    for col in all_purpose_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0
    new_columns = {
        group_col: df_copy[cols].sum(axis=1).astype(int)
        for group_col, cols in purpose_groups.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(new_columns, index=df_copy.index)], axis=1)

    for term in [36, 60]:
        df_copy[f'term:{term}'] = (df_copy['term_int'] == term).astype(int)

    emp_length_buckets = {
        'emp_length:0': [0],
        'emp_length:1': [1],
        'emp_length:2-4': range(2, 5),
        'emp_length:5-6': range(5, 7),
        'emp_length:7-9': range(7, 10),
        'emp_length:10': [10]
    }
    emp_length_cols = {
        label: df_copy['emp_length_int'].isin(values).astype(int)
        for label, values in emp_length_buckets.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(emp_length_cols, index=df_copy.index)], axis=1)

    mths_issue_buckets = {
        'mths_since_issue_d:<38': range(0, 38),
        'mths_since_issue_d:38-39': range(38, 40),
        'mths_since_issue_d:40-41': range(40, 42),
        'mths_since_issue_d:42-48': range(42, 49),
        'mths_since_issue_d:49-52': range(49, 53),
        'mths_since_issue_d:53-64': range(53, 65),
        'mths_since_issue_d:65-84': range(65, 85),
        'mths_since_issue_d:>84': range(85, int(df_copy['mths_since_issue_d'].max()) + 1)
    }
    mths_issue_cols = {
        label: df_copy['mths_since_issue_d'].isin(value_range).astype(int)
        for label, value_range in mths_issue_buckets.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(mths_issue_cols, index=df_copy.index)], axis=1)

    int_rate_bins = {
        'int_rate:<9.548': lambda x: x <= 9.548,
        'int_rate:9.548-12.025': lambda x: (x > 9.548) & (x <= 12.025),
        'int_rate:12.025-15.74': lambda x: (x > 12.025) & (x <= 15.74),
        'int_rate:15.74-20.281': lambda x: (x > 15.74) & (x <= 20.281),
        'int_rate:>20.281': lambda x: x > 20.281
    }
    int_rate_cols = {
        label: condition(df_copy['int_rate']).astype(int)
        for label, condition in int_rate_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(int_rate_cols, index=df_copy.index)], axis=1)

    mths_earliest_bins = {
        'mths_since_earliest_cr_line:<140': lambda x: x < 140,
        'mths_since_earliest_cr_line:141-164': lambda x: (x >= 140) & (x <= 164),
        'mths_since_earliest_cr_line:165-247': lambda x: (x >= 165) & (x <= 247),
        'mths_since_earliest_cr_line:248-270': lambda x: (x >= 248) & (x <= 270),
        'mths_since_earliest_cr_line:271-352': lambda x: (x >= 271) & (x <= 352),
        'mths_since_earliest_cr_line:>352': lambda x: x > 352
    }
    new_columns = {
        label: condition(df_copy['mths_since_earliest_cr_line']).astype(int)
        for label, condition in mths_earliest_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(new_columns, index=df_copy.index)], axis=1)

    delinq_2yrs_bins = {
        'delinq_2yrs:0': lambda x: x == 0,
        'delinq_2yrs:1-3': lambda x: (x >= 1) & (x <= 3),
        'delinq_2yrs:>=4': lambda x: x >= 4
    }
    new_columns = {
        label: condition(df_copy['delinq_2yrs']).astype(int)
        for label, condition in delinq_2yrs_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(new_columns, index=df_copy.index)], axis=1)

    inq_last_6mths_bins = {
        'inq_last_6mths:0': lambda x: x == 0,
        'inq_last_6mths:1-2': lambda x: (x >= 1) & (x <= 2),
        'inq_last_6mths:3-6': lambda x: (x >= 3) & (x <= 6),
        'inq_last_6mths:>6': lambda x: x > 6
    }
    inq_last_6mths_cols = {
        label: condition(df_copy['inq_last_6mths']).astype(int)
        for label, condition in inq_last_6mths_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(inq_last_6mths_cols, index=df_copy.index)], axis=1)

    open_acc_bins = {
        'open_acc:0': lambda x: x == 0,
        'open_acc:1-3': lambda x: (x >= 1) & (x <= 3),
        'open_acc:4-12': lambda x: (x >= 4) & (x <= 12),
        'open_acc:13-17': lambda x: (x >= 13) & (x <= 17),
        'open_acc:18-22': lambda x: (x >= 18) & (x <= 22),
        'open_acc:23-25': lambda x: (x >= 23) & (x <= 25),
        'open_acc:26-30': lambda x: (x >= 26) & (x <= 30),
        'open_acc:>=31': lambda x: x >= 31
    }
    open_acc_cols = {
        label: condition(df_copy['open_acc']).astype(int)
        for label, condition in open_acc_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(open_acc_cols, index=df_copy.index)], axis=1)

    pub_rec_bins = {
        'pub_rec:0-2': lambda x: (x >= 0) & (x <= 2),
        'pub_rec:3-4': lambda x: (x >= 3) & (x <= 4),
        'pub_rec:>=5': lambda x: x >= 5
    }
    pub_rec_cols = {
        label: condition(df_copy['pub_rec']).astype(int)
        for label, condition in pub_rec_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(pub_rec_cols, index=df_copy.index)], axis=1)

    total_acc_bins = {
        'total_acc:<=27': lambda x: x <= 27,
        'total_acc:28-51': lambda x: (x >= 28) & (x <= 51),
        'total_acc:>=52': lambda x: x >= 52
    }
    total_acc_cols = {
        label: condition(df_copy['total_acc']).astype(int)
        for label, condition in total_acc_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(total_acc_cols, index=df_copy.index)], axis=1)

    acc_now_delinq_bins = {
        'acc_now_delinq:0': lambda x: x == 0,
        'acc_now_delinq:>=1': lambda x: x >= 1
    }
    acc_now_delinq_cols = {
        label: condition(df_copy['acc_now_delinq']).astype(int)
        for label, condition in acc_now_delinq_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(acc_now_delinq_cols, index=df_copy.index)], axis=1)

    total_rev_hi_lim_bins = {
        'total_rev_hi_lim:<=5K': lambda x: x <= 5000,
        'total_rev_hi_lim:5K-10K': lambda x: (x > 5000) & (x <= 10000),
        'total_rev_hi_lim:10K-20K': lambda x: (x > 10000) & (x <= 20000),
        'total_rev_hi_lim:20K-30K': lambda x: (x > 20000) & (x <= 30000),
        'total_rev_hi_lim:30K-40K': lambda x: (x > 30000) & (x <= 40000),
        'total_rev_hi_lim:40K-55K': lambda x: (x > 40000) & (x <= 55000),
        'total_rev_hi_lim:55K-95K': lambda x: (x > 55000) & (x <= 95000),
        'total_rev_hi_lim:>95K': lambda x: x > 95000
    }
    total_rev_hi_lim_cols = {
        label: condition(df_copy['total_rev_hi_lim']).astype(int)
        for label, condition in total_rev_hi_lim_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(total_rev_hi_lim_cols, index=df_copy.index)], axis=1)

    annual_inc_bins = {
        'annual_inc:<20K': lambda x: x <= 20000,
        'annual_inc:20K-30K': lambda x: (x > 20000) & (x <= 30000),
        'annual_inc:30K-40K': lambda x: (x > 30000) & (x <= 40000),
        'annual_inc:40K-50K': lambda x: (x > 40000) & (x <= 50000),
        'annual_inc:50K-60K': lambda x: (x > 50000) & (x <= 60000),
        'annual_inc:60K-70K': lambda x: (x > 60000) & (x <= 70000),
        'annual_inc:70K-80K': lambda x: (x > 70000) & (x <= 80000),
        'annual_inc:80K-90K': lambda x: (x > 80000) & (x <= 90000),
        'annual_inc:90K-100K': lambda x: (x > 90000) & (x <= 100000),
        'annual_inc:100K-120K': lambda x: (x > 100000) & (x <= 120000),
        'annual_inc:120K-140K': lambda x: (x > 120000) & (x <= 140000),
        'annual_inc:>140K': lambda x: x > 140000
    }
    annual_inc_cols = {
        label: condition(df_copy['annual_inc']).astype(int)
        for label, condition in annual_inc_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(annual_inc_cols, index=df_copy.index)], axis=1)

    missing_col = {
        'mths_since_last_delinq:Missing': df_copy['mths_since_last_delinq'].isnull().astype(int)
    }
    delinq_bins = {
        'mths_since_last_delinq:0-3': lambda x: (x >= 0) & (x <= 3),
        'mths_since_last_delinq:4-30': lambda x: (x >= 4) & (x <= 30),
        'mths_since_last_delinq:31-56': lambda x: (x >= 31) & (x <= 56),
        'mths_since_last_delinq:>=57': lambda x: x >= 57
    }
    filled_values = df_copy['mths_since_last_delinq'].fillna(-1)
    delinq_cols = {
        label: condition(filled_values).astype(int)
        for label, condition in delinq_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame({**missing_col, **delinq_cols}, index=df_copy.index)], axis=1)

    dti_bins = {
        'dti:<=1.4': lambda x: x <= 1.4,
        'dti:1.4-3.5': lambda x: (x > 1.4) & (x <= 3.5),
        'dti:3.5-7.7': lambda x: (x > 3.5) & (x <= 7.7),
        'dti:7.7-10.5': lambda x: (x > 7.7) & (x <= 10.5),
        'dti:10.5-16.1': lambda x: (x > 10.5) & (x <= 16.1),
        'dti:16.1-20.3': lambda x: (x > 16.1) & (x <= 20.3),
        'dti:20.3-21.7': lambda x: (x > 20.3) & (x <= 21.7),
        'dti:21.7-22.4': lambda x: (x > 21.7) & (x <= 22.4),
        'dti:22.4-35': lambda x: (x > 22.4) & (x <= 35),
        'dti:>35': lambda x: x > 35
    }
    dti_bin_columns = {
        col_name: condition(df_copy['dti']).astype(int)
        for col_name, condition in dti_bins.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(dti_bin_columns, index=df_copy.index)], axis=1)

    df_copy['mths_since_last_record:Missing'] = df_copy['mths_since_last_record'].isnull().astype(int)
    filled_record = df_copy['mths_since_last_record'].fillna(-1)
    record_ranges = {
        'mths_since_last_record:0-2': (0, 2),
        'mths_since_last_record:3-20': (3, 20),
        'mths_since_last_record:21-31': (21, 31),
        'mths_since_last_record:32-80': (32, 80),
        'mths_since_last_record:81-86': (81, 86),
        'mths_since_last_record:>86': (87, float('inf'))
    }
    record_bin_columns = {
        label: ((filled_record >= low) & (filled_record <= high)).astype(int)
        for label, (low, high) in record_ranges.items()
    }
    df_copy = pd.concat([df_copy, pd.DataFrame(record_bin_columns, index=df_copy.index)], axis=1)

    return df_copy

def select_model_variables(dataframe, variable_list):
    selected_df = dataframe.loc[:, variable_list]
    return selected_df

class LogisticRegression_with_p_values:

    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

class LinearRegression(SklearnLinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)

    def fit(self, X, y):
        super().fit(X, y)
        predictions = self.predict(X)
        residuals = y - predictions
        df = X.shape[0] - X.shape[1] - int(self.fit_intercept)
        sse = np.sum(residuals ** 2) / df
        X_design = np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X
        var_b = sse * np.linalg.inv(X_design.T @ X_design)
        se_b = np.sqrt(np.diag(var_b))
        self.t = self.coef_ / se_b[1:] if self.fit_intercept else self.coef_ / se_b
        self.p = 2 * (1 - stat.t.cdf(np.abs(self.t), df))

        return self



def adjust_scorecard(df):
    df = df.copy()

    # Step 1: Calculate the difference between preliminary and calculation
    df['Difference'] = df['Score - Preliminary'] - df['Score - Calculation']

    # Step 2: Initialize 'Score - Final' as equal to 'Score - Preliminary'
    df['Score - Final'] = df['Score - Preliminary']

    # Step 3: Find the index of the largest rounding difference
    idx_max_diff = df['Difference'].idxmax()

    # Step 4: Adjust that score to floor (rounded down)
    df.at[idx_max_diff, 'Score - Final'] = int(df.at[idx_max_diff, 'Score - Calculation'] // 1)

    return df