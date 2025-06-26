def clean_data(df):
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1]
    return df
