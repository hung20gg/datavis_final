from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 

current_dir = os.path.dirname(__file__)

def analyze_feat(series):
    
    # Check the type of the feature
    type_ = series.dtype
    if type_ != 'object':
        
    
        # Check the skewness of the feature
        skew = series.skew()
        
        # Check the kurtosis of the feature
        kurt = series.kurt()
        
        # Check the missing values of the feature
        missing = series.isnull().sum()
        
        # Can convert to interger?
        is_interger = 0
        
        # Min, max, mean, median, std, unique values
        min_ = series.min()
        max_ = series.max()
        mean_ = series.mean()
        median_ = series.median()
        std_ = series.std()
    else:
        skew = 0
        kurt = 0
        missing = 0
        is_interger = False
        min_ = 0
        max_ = 0
        mean_ = 0
        median_ = 0
        std_ = 0   
        
    unique_ = series.nunique()
    
    # Return the results as a row 
    return (type_, skew, kurt, missing, is_interger, min_, max_, mean_, median_, std_, unique_)

def analyze_column(feature, df):
    # Get the results for a specific feature
    return [feature] + list(analyze_feat(df[feature]))
   
def analyze_df(df, n_jobs):
    features = df.columns
    if 'TARGET' in features:
        df.drop(columns=['TARGET'], inplace=True)
        features = df.columns
    # Create a DataFrame to store the results
    
    
    if not isinstance(n_jobs, int):
        results = []
        # Loop through each feature
        for feature in tqdm(features):
            # Get the results
            result = analyze_feat(df[feature])
            
            # Append the results as a new row
            results.append([feature] + list(result))
                
    else:
        results = []

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            # Use tqdm to display the progress bar
            futures = [executor.submit(analyze_column, feature, df) for feature in features]
            for future in futures:
                result = future.result()  # Get the result from the future
                results.append(result)
    
    return pd.DataFrame(results, columns=['Feature', 'Type', 'Skewness', 'Kurtosis', 'Missing', 'Is Integer', 'Min', 'Max', 'Mean', 'Median', 'Std', 'Unique'])
    
def pca_feat(df):
    df_num = df.select_dtypes(include=[np.number])
    df_num.fillna(0, inplace=True)
    
    pca = PCA()
    scaler = StandardScaler()
    
    df_num_scaled = scaler.fit_transform(df_num)
    pca.fit_transform(df_num_scaled)
    
    explained_variance = pca.explained_variance_ratio_

    # Get the absolute values of PCA components (loadings)
    feature_contributions = np.abs(pca.components_)

    # Aggregate feature importance scores
    # Weight each feature's contribution by the variance explained by the component
    weighted_contributions = feature_contributions * explained_variance[:, np.newaxis]

    # Sum weighted contributions across components
    feature_importance = weighted_contributions.sum(axis=0)

    # Create a DataFrame for ranking
    importance_df = pd.DataFrame({
        'Feature': df_num.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    return importance_df
    
    
    
def check_feature(df):
    # Get the features
    df_result = analyze_df(df)  
    # Get the PCA feature importance
    pca_results = pca_feat(df)
    
    df_result = df_result.merge(pca_results, on='Feature', how='left')
    df_result.fillna(0, inplace=True)
    
    return df_result

def power_scaler_col(df, n_jobs = None, skewness = 8, kurtosis = 50, use_cache = True):
    # Return which columns for power transformation, which for standard scaler
    
    cache_path = os.path.join(current_dir, f'../temp/feature_analysis_{df.shape[0]}_{df.shape[1]}.csv')
    if os.path.exists(cache_path) and use_cache:
        print('Cache found')
        df_result = pd.read_csv(cache_path)
    else:

        df_num = df.select_dtypes(include=[np.number])
        df_num.fillna(0, inplace=True)
        df_result = analyze_df(df, n_jobs)
        df_result.to_csv(cache_path, index=False)

    # Nunique < 100 -> Categorical feature -> Scale
    df_result['Mask_nunique'] = df_result['Unique'] > 100
    
    # Skewness > 10 -> Power transformation
    df_result['Mask_skew'] = df_result['Skewness'].abs() > skewness
    
    # Kurtosis > 15 -> Power transformation
    df_result['Mask_kurt'] = (df_result['Kurtosis']-3).abs() > kurtosis
    
    # Missing > 9/10 dataset -> Scale
    df_result['Mask_missing'] = df_result['Missing'] < 0.5 * df.shape[0]
    
    df_result['Mask'] = df_result['Mask_nunique'] & (df_result['Mask_skew'] | df_result['Mask_kurt']) & df_result['Mask_missing']
    
    df_result['Min_Max'] = ( df_result['Min'] == 0) & (df_result['Max'] < 20) & (df_result['Unique'] < 50) # 50/50
    
    df_result
    
    power_col = df_result[df_result['Mask'] == 1].Feature.tolist()
    standard_col = df_result[(df_result['Mask'] == 0) & (df_result['Min_Max'] == 0)].Feature.tolist()
    min_max_col = df_result[(df_result['Mask'] == 0) & (df_result['Min_Max'] == 1)].Feature.tolist()
    
    
    return power_col, standard_col, min_max_col

    
def exp_scaler_col(df, n_jobs = None):
    # Return which columns for power transformation, which for standard scaler
    
    df_num = df.select_dtypes(include=[np.number])
    df_num.fillna(0, inplace=True)
    
    df_result = analyze_df(df, n_jobs)
    
    # Nunique < 100 -> Categorical feature -> Scale
    df_result['Mask_nunique'] = df_result['Unique'] > 100
    
    # Skewness > 10 -> Power transformation
    df_result['Mask_skew'] = df_result['Skewness'].abs() > 5
    
    # # Kurtosis > 15 -> Power transformation
    # df_result['Mask_kurt'] = (df_result['Kurtosis']-3).abs() > 50
    
    # Missing > 9/10 dataset -> Scale
    df_result['Mask_missing'] = df_result['Missing'] < 0.5 * df.shape[0]
    
    df_result['Mask'] = df_result['Mask_nunique'] & df_result['Mask_skew'] & df_result['Mask_missing']
    
    df_result['Min_Max'] = ( df_result['Min'] == 0) & (df_result['Max'] < 50) & (df_result['Unique'] < 50)
    
    df_result
    
    power_col = df_result[df_result['Mask'] == 1].Feature.tolist()
    standard_col = df_result[(df_result['Mask'] == 0) & (df_result['Min_Max'] == 0)].Feature.tolist()
    min_max_col = df_result[(df_result['Mask'] == 0) & (df_result['Min_Max'] == 1)].Feature.tolist()
    
    
    return power_col, standard_col, min_max_col


if __name__ == '__main__':
    df = pd.read_parquet('../data/df_train.parquet')
    result = check_feature(df)
    print(result)
    # result.to_csv('../data/feature_analysis.csv', index=False)
    result.to_excel('../data/feature_analysis.xlsx', index=False)