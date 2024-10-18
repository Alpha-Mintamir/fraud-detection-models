import pandas as pd
import os
import logging
import ipaddress
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np



# Logging configuration
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)




def data_loader(path):
    """Load data from a CSV file."""
    logger.info(f'Loading data from {path}')
    data = pd.read_csv(path)
    logger.info('Data loaded successfully')
    return data

def missing_values_table(df):
    """Generate a table of missing values and their percentages."""
    logger.info('Generating missing values table')
    
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_data_types = df.dtypes
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)
    
    logger.info(f"DataFrame has {df.shape[1]} columns.")
    logger.info(f"{mis_val_table_ren_columns.shape[0]} columns have missing values.")
    
    return mis_val_table_ren_columns


def column_summary(df):
    """Generate a summary of columns in the DataFrame."""
    logger.info('Generating column summary')
    summary_data = []
    
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()
        
        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info('Column summary generated successfully')
    return summary_df

def impute_missing_values(df: pd.DataFrame, column: str, method: str = 'mean') -> pd.DataFrame:
    """Impute missing values in the specified column of a DataFrame."""
    logger.info(f'Imputing missing values in column: {column} using method: {method}')
    
    if method not in ['mean', 'median', 'mode']:
        logger.error("Invalid imputation method provided.")
        raise ValueError("Method must be 'mean', 'mode' or 'median'")
    
    if method == 'mean':
        value = df[column].mean()
    elif method == 'median':
        value = df[column].median()
    elif method == 'mode':
        value = df[column].mode()[0]
    
    df[column].fillna(value, inplace=True)
    logger.info(f'Missing values imputed in column: {column}')
    
    return df

def impute_with_historical_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear'."""
    logger.info('Imputing missing values with historical averages')
    
    mode_month = df['CompetitionOpenSinceMonth'].mode()[0]
    mode_year = df['CompetitionOpenSinceYear'].mode()[0]
    
    df['CompetitionOpenSinceMonth'].fillna(mode_month, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(mode_year, inplace=True)
    
    logger.info('Missing values imputed with historical averages')
    return df


def ip_to_int(ip):
    """
    Convert an IP address string to its integer equivalent.
    """
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        logger.error(f"Invalid IP address encountered: {ip}")
        return None

def merge_datasets_for_geolocation(df_fraud, df_ip_country):
    """
    Merge Fraud_Data.csv and IpAddress_to_Country.csv for geolocation analysis.

    Parameters:
    df_fraud (pd.DataFrame): DataFrame containing fraud transaction data with 'ip_address' column.
    df_ip_country (pd.DataFrame): DataFrame containing IP address ranges and country data.

    Returns:
    pd.DataFrame: Merged DataFrame with geolocation (country) information based on IP address.
    """
    logger.info('Starting merge of datasets for geolocation analysis')

    try:
        # Convert IP address in Fraud_Data.csv to integer format
        logger.info('Converting IP addresses in fraud data to integer format')
        df_fraud['ip_address_int'] = df_fraud['ip_address'].apply(ip_to_int)

        # Convert lower and upper bound IP addresses in IpAddress_to_Country.csv to integer format
        logger.info('Converting IP address bounds in country data to integer format')
        df_ip_country['lower_bound_ip_address_int'] = df_ip_country['lower_bound_ip_address'].apply(ip_to_int)
        df_ip_country['upper_bound_ip_address_int'] = df_ip_country['upper_bound_ip_address'].apply(ip_to_int)

        # Drop rows with invalid IP addresses (None values after conversion)
        logger.info('Dropping rows with invalid IP addresses')
        df_fraud.dropna(subset=['ip_address_int'], inplace=True)
        df_ip_country.dropna(subset=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)

        # Merge: For each IP address in fraud data, find where it falls within the lower and upper bounds in IP address to country data
        logger.info('Merging datasets based on IP address ranges')
        merged_df = pd.merge_asof(
            df_fraud.sort_values('ip_address_int'),  # Sort by IP address for merge_asof to work
            df_ip_country.sort_values('lower_bound_ip_address_int'),  # Sort by lower bound IP for merge_asof
            left_on='ip_address_int',  # Column in Fraud_Data.csv to match
            right_on='lower_bound_ip_address_int',  # Column in IpAddress_to_Country.csv to match
            direction='backward',  # Find closest matching lower_bound_ip_address less than or equal to the ip_address_int
            suffixes=('_fraud', '_country')
        )

        # Filter the rows where the 'ip_address_int' is within the IP range (lower_bound_ip_address and upper_bound_ip_address)
        logger.info('Filtering merged data to ensure IP address is within the specified range')
        merged_df = merged_df[
            (merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) &
            (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int'])
        ]

        logger.info('Merge completed successfully')
        return merged_df

    except Exception as e:
        logger.error(f"An error occurred during merging: {e}")
        return df_fraud
    





def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded categorical features
    """
    logger.info('Encoding categorical features using one-hot encoding')
    df_copy = df.copy()
    
    categorical_columns = df_copy.select_dtypes(include=['object']).columns
    exclude_columns = ['ip_address', 'user_id', 'device_id']
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df_copy, columns=categorical_columns, drop_first=True)
    
    logger.info('Categorical features encoded successfully')
    return df_encoded

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features using StandardScaler
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with normalized features
    """
    logger.info('Normalizing numerical features using StandardScaler')
    df_copy = df.copy()
    scaler = StandardScaler()
    
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    exclude_columns = ['class', 'Class', 'user_id', 'device_id']
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    df_copy[numeric_columns] = scaler.fit_transform(df_copy[numeric_columns])
    
    logger.info('Numerical features normalized successfully')
    return df_copy