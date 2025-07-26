# Import Libraries
import streamlit as st
import datetime
import pandas as pd
import numpy as np
from joblib import load

# Import Functions
from functions import select_model_variables, preprocess_inputs_pd, preprocess_inputs_lgd

# Import Models
from model import reg_lgd_st_1, reg_lgd_st_2

# Import Variables
from model_training_variables import pd_variable_list, pd_ref_categories, reference_date, lgd_features_all, lgd_features_reference_cat

# Import Scorecard Data Frame
from model import df_scorecard

@st.cache_resource(show_spinner="Loading PD model...")
def load_pd_model():
    return load("pd_model.joblib")

@st.cache_resource(show_spinner="Loading LGD stage 1 model...")
def load_lgd_1_model():
    return load("lgd_stage_1_model.joblib")

@st.cache_resource(show_spinner="Loading LGD stage 2 model...")
def load_lgd_2_model():
    return load("lgd_stage_2_model.joblib")

@st.cache_resource(show_spinner="Loading EAD stage 1 model...")
def load_ead_model():
    return load("ead_model.joblib")

if __name__ == '__main__':
    #Input Form
    st.set_page_config(layout="wide")

    # Initialize Session State Dataframes
    if 'upload' not in st.session_state:
        st.session_state.upload = None

    if 'upload_pd_w_scores' not in st.session_state:
        st.session_state.upload_pd_w_scores = None

    if 'filtered_pd' not in st.session_state:
        st.session_state.filtered_pd = None

    with st.form('data_link'):
        st.write("Connect Your Target Portfolio Data")
        portfolio_url = st.text_input("Enter a Data URL", placeholder="https://example.com")
        data_link = st.form_submit_button("Upload Portfolio Data")

        if data_link:
            upload = pd.read_csv(portfolio_url)
            st.session_state.upload = upload

            upload_pd = preprocess_inputs_pd(upload, reference_date)

            # Produces credit scores for the portfolio
            user_input_with_ref_cat = select_model_variables(upload_pd, pd_variable_list) #This will the input file from the user
            user_input_with_ref_cat_w_intercept = user_input_with_ref_cat
            user_input_with_ref_cat_w_intercept.insert(0,'Intercept',1)
            user_input_with_ref_cat_w_intercept = user_input_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
            scorecard_scores = df_scorecard['Score - Final']
            scorecard_scores = scorecard_scores.values.reshape(102,1)
            all_scores = user_input_with_ref_cat_w_intercept.dot(scorecard_scores)
            all_scores.columns = ['credit_score']
            upload_pd_w_scores = pd.concat([upload_pd,all_scores], axis=1)
            st.session_state.upload_pd_w_scores = upload_pd_w_scores
            st.success("✅ Data Loaded")


    with st.form('credit_score_filter'):
        filtered_pd = st.session_state.upload_pd_w_scores
        min_credit_score = st.number_input("Minimum Credit Score", min_value=300, max_value=850, value=720, step=10)
        filter = st.form_submit_button("Filter Portfolio")

        if filter:
            filtered_pd = filtered_pd[filtered_pd['credit_score'] >= min_credit_score]
            st.session_state.filtered_pd = filtered_pd
            st.success("✅ Data Filtered")

    with st.form('exp_loss_pred'):
        predict = st.form_submit_button("Predict Expected Credit Loss")

        if predict:

            loan_data_inputs_pd = st.session_state.filtered_pd

            loan_data_lgd_ead = preprocess_inputs_lgd(st.session_state.upload, reference_date)
            loan_data_lgd_ead = loan_data_lgd_ead.loc[loan_data_lgd_ead.index.isin(loan_data_inputs_pd.index)]
            loan_data_preprocessed_lgd_ead = loan_data_lgd_ead[lgd_features_all]
            loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(lgd_features_reference_cat, axis=1)

            reg_pd = load_pd_model()
            reg_lgd_st_1 = load_lgd_1_model()
            reg_lgd_st_2 = load_lgd_2_model()
            reg_ead = load_ead_model()

            #Predict LGD Stage 1 and 2
            loan_data_lgd_ead['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(loan_data_preprocessed_lgd_ead)
            loan_data_lgd_ead['recovery_rate_st_2'] = reg_lgd_st_2.predict(loan_data_preprocessed_lgd_ead)
            loan_data_lgd_ead['recovery_rate'] = loan_data_lgd_ead['recovery_rate_st_1'] * loan_data_lgd_ead['recovery_rate_st_2']
            loan_data_lgd_ead['recovery_rate'] = np.where(loan_data_lgd_ead['recovery_rate'] < 0, 0, loan_data_lgd_ead['recovery_rate'])
            loan_data_lgd_ead['recovery_rate'] = np.where(loan_data_lgd_ead['recovery_rate'] > 1, 1, loan_data_lgd_ead['recovery_rate'])
            loan_data_lgd_ead['LGD'] = 1 - loan_data_lgd_ead['recovery_rate']

            # Predict EAD
            loan_data_lgd_ead['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)
            loan_data_lgd_ead['CCF'] = np.where(loan_data_lgd_ead['CCF'] < 0, 0, loan_data_lgd_ead['CCF'])
            loan_data_lgd_ead['CCF'] = np.where(loan_data_lgd_ead['CCF'] > 1, 1, loan_data_lgd_ead['CCF'])
            loan_data_lgd_ead['EAD'] = loan_data_lgd_ead['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

            # Predict PD
            loan_data_inputs_pd_temp = loan_data_inputs_pd[pd_variable_list]
            loan_data_inputs_pd_temp = loan_data_inputs_pd_temp.drop(pd_ref_categories, axis=1)
            loan_data_inputs_pd['PD'] = reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[:][:, 0]
            loan_data_preprocessed_new = pd.concat([loan_data_lgd_ead, loan_data_inputs_pd], axis=1)

            # End Result
            loan_data_preprocessed_new['EL'] = loan_data_preprocessed_new['PD'] * loan_data_preprocessed_new['LGD'] * loan_data_preprocessed_new['EAD']
            loan_data_preprocessed_new = loan_data_preprocessed_new.loc[:, ~loan_data_preprocessed_new.columns.duplicated(keep='first')]

            # Expected Loss Components and Expected Loss
            # Export this CSV
            loan_data_preprocessed_new[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head()

            sum_exp_loss = loan_data_preprocessed_new['EL'].sum()
            sum_funded_amt = loan_data_preprocessed_new['funded_amnt'].sum()

            formatted_sum_exp_loss = f"{sum_exp_loss:,.2f}"
            formatted_sum_funded_amt = f"{sum_funded_amt:,.2f}"

            st.write(f"Sum of Expected Loss: $ {formatted_sum_exp_loss}")
            st.write(f"Portfolio Funded Amount: $ {formatted_sum_funded_amt}")
            st.write(f"Expected Credit Loss: {(sum_exp_loss/sum_funded_amt):.2%}")
            st.write(loan_data_preprocessed_new[['funded_amnt', 'PD', 'LGD', 'EAD', 'EL']].head())

