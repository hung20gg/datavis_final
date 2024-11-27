def analyze_feat(series):
    
    # Check the type of the feature
    type_ = series.dtype
    
    # Check the skewness of the feature
    skew = series.skew()
    
    # Check the kurtosis of the feature
    kurt = series.kurt()
    
    # Check the missing values of the feature
    missing = series.isnull().sum()
    
    # Can convert to interger?
    is_interger = series.apply(lambda x: x.is_integer()).all()
    
    # Min, max, mean, median, std, unique values
    min_ = series.min()
    max_ = series.max()
    mean_ = series.mean()
    median_ = series.median()
    std_ = series.std()
    unique_ = series.unique()
    
    # Return the results as a row 
    return (type_, skew, kurt, missing, is_interger, min_, max_, mean_, median_, std_, unique_)
    