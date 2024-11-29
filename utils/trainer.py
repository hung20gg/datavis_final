import warnings
warnings.filterwarnings('ignore')

import sys 
import os
sys.path.append('..')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

from utils.check_feature import analyze_df, power_scaler_col

import argparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, OrdinalEncoder, LabelEncoder, MinMaxScaler

# Define the argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse parameters for the script.")
    
    # Add arguments
    parser.add_argument(
        "-t", 
        type=bool, 
        default=True, 
        help="Enable transform first (default: True)"
    )
    parser.add_argument(
        "-u", 
        type=float, 
        default=0.25, 
        help="Upsample ratio (default: 0.25)"
    )
    parser.add_argument(
        "-k", 
        type=bool, 
        default=False, 
        help="Retrain KNN (default: False)"
    )
    
    return parser.parse_args()


def setup_preprocess_pipeline(cate_cols, power_col, ordinal_cols, standard_col, min_max_col):
    onehot_transformer = OneHotEncoder(handle_unknown='ignore')
    power_transformer = PowerTransformer()
    ordinal_transformer = OrdinalEncoder()
    scaler_transformer = StandardScaler()
    min_max_transformer = MinMaxScaler()

    return ColumnTransformer(
        transformers=[
            ('onehot', onehot_transformer, cate_cols),
            ('power', power_transformer, power_col), # power_transformer
            ('ordinal', ordinal_transformer, ordinal_cols),
            ('scale', scaler_transformer, standard_col),
            ('min_max', min_max_transformer, min_max_col)
            
        ]
    )



def load_clean_process_data():
    # Load the data
    X = pd.read_parquet("../data/df_train.parquet")
    
    # Display the first few rows
    y = X['TARGET']
    X.drop(columns=['TARGET', 'SK_ID_CURR'], inplace=True)
    
    
    ordinal_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE', 'CODE_GENDER', 'NAME_CONTRACT_TYPE']
    num_cols = []
    cate_cols = []
    flag_cols = []
    for col in X.columns:
        if col not in ordinal_cols:
            if X[col].dtype == 'float64':
                num_cols.append(col)
            elif X[col].dtype == 'int64':
                num_cols.append(col)
            else:
                cate_cols.append(col)
    
    power_col, standard_col, min_max_col  = power_scaler_col(X[num_cols], n_jobs=8)
    X[num_cols] = X[num_cols].clip(-999999999, 999999999)
    
    for col in num_cols:
        X[col].fillna(0, inplace=True)
    
    for col in cate_cols:
        X[col].fillna('Unknown', inplace=True)
    
    return X, y

# Main function
if __name__ == "__main__":
    args = parse_arguments()
    
    # Access parsed arguments
    TRANSFORM_FIRST = args.Transform
    UPSAMPLE_RATIO = args.Upsample
    RETRAIN_KNN = args.RetrainKNN
    
    # Display parsed arguments
    print("\nParsed Parameters:")
    print(f"TRANSFORM_FIRST: {TRANSFORM_FIRST}")
    print(f"UPSAMPLE_RATIO: {UPSAMPLE_RATIO}")
    print(f"RETRAIN_KNN: {RETRAIN_KNN}")