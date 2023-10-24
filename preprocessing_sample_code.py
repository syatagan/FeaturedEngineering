import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def drop_duplicates(df, subset_name):
    df.drop_duplicates(subset=[subset_name], inplace=True)
    return df

def encode(df, column_to_encode):
    le = LabelEncoder()
    # fit and transform a column using the LabelEncoder
    df[column_to_encode] = le.fit_transform(df[column_to_encode])
    return df

def outlier_handling(df, column_with_outliers):
    q1 = df[column_with_outliers].quantile(0.25)
    q3 = df[column_with_outliers].quantile(0.75)
    iqr = q3 - q1
    # remove outliers
    df = df[(df[column_with_outliers] > (q1 - 1.5 * iqr))
            & (df[column_with_outliers] < (q3 + 1.5 * iqr))]
    return df


def date_formatting(df, column_with_date):
    # format date column
    df[column_with_date] = pd.to_datetime(df[column_with_date],
                                          format='%m/%d/%Y')
    return df


def remove_missing_values(df):
    # Find missing values
    missing_values = df.isnull().sum()
    # Remove rows with missing values
    df = df.dropna()
    # Print number of missing values removed
    print("Removed {} missing values".format(missing_values.sum()))
    return df


def data_cleaning_pipeline(df_path,
                           duplication_subset,
                           column_to_encode,
                           column_with_outliers,
                           column_with_date):
    df = pd.read_csv(df_path)
    df_no_duplicates = drop_duplicates(df, duplication_subset)
    df_encoded = encode(df_no_duplicates, column_to_encode)
    df_no_outliers = outlier_handling(df_encoded, column_with_outliers)
    df_date_formatted = date_formatting(df_no_outliers, column_with_date)
    df_no_nulls = remove_missing_values(df_date_formatted)
    return df_no_nulls


# Create a sample DataFrame
data = {'Name': ['John', 'Jane', 'Bob', 'John', 'Alice'],
        'Age': [30, 25, 40, 30, np.NaN],
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
        'Income': [50000, 60000, 70000, 45000, 80000],
        'Birthdate': ['01/01/1990', '02/14/1996', '03/15/1981',
                      '01/01/1990', '06/30/1986'],
        'Married': [True, False, True, False, True],
        'Children': [2, 0, 1, 0, 3]}
df = pd.DataFrame(data)
print('Before Preprocessing:\n', df)
# Save DataFrame as CSV file
df.to_csv('my_data.csv', index=False)

clean_df = data_cleaning_pipeline('my_data.csv',
                                  'Name',
                                  'Gender',
                                  'Income',
                                  'Birthdate')

print('\nAfter preprocessing')
clean_df.head()
