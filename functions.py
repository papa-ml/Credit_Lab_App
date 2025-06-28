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

def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()

    return df