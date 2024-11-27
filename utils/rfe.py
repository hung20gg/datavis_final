from multiprocessing import Pool 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, make_scorer, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

def metric(type_, *args):
    if type_ == 'auc':
        return roc_auc_score(*args)
    if type_ == 'accuracy':
        return accuracy_score(*args)
    if type_ == 'gini':
        return 2*roc_auc_score(*args) - 1
    if type_ == 'f1':
        return f1_score(*args)
    raise ValueError(f"Unknown metric type: {type_}")

def train(X_train, y_train): # -> model
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=1)
    lr.fit(X_train, y_train) 

    return lr 


def evaluate(X_train, y_train, X_test, y_test, mask, scorer = 'gini', index = None):
    lr = train(X_train[: , mask], y_train)
    y_pred = lr.predict(X_test[: , mask])

    if index is None:
        return metric(scorer, y_test, y_pred)
    return index, metric(scorer, y_test, y_pred)

def chop_feature_parallel(X_train, y_train, X_test, y_test, scorer, orginal_mask, skip):
    masks = []
    pos = []
    features = len(orginal_mask)

    for i in range(features):
        
        if orginal_mask[i] == True:
            pos.append(i)
            mask2 = orginal_mask.copy()
            mask2[i] = False
            mask2 |= skip
            masks.append(mask2)

    with Pool(os.cpu_count()) as p:
        scores = p.starmap(evaluate, [(X_train, y_train, X_test, y_test, mask, scorer, i) for i, mask in zip(pos, masks)])
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[-1] # [(index, score), ...]


def RFE(X_train, y_train, X_test, y_test, scorer, n_features_to_select, skip=None, columns = None):
    n_features = X_train.shape[1]
    scores = []
    least_important_features = []
    
    # Skip must be a mask:

    selected_features = np.ones(n_features, dtype=bool)  

    if skip is None:
        skip = np.zeros(n_features, dtype=bool)

    else:
        selected_features &= ~skip

    for i in tqdm(range(n_features - n_features_to_select)):
        
        # Less important feat has different size, so we need current_features to get the correct index
        index, score = chop_feature_parallel(X_train, y_train, X_test, y_test, scorer, selected_features, skip)

        selected_features[index] = False
        print(selected_features)
        # Update current_features
        
        scores.append(score)
        least_important_features.append(index)
        
        if columns is not None:
            print(f"Removed feature: {i+1} ", columns[index])
    return least_important_features, scores


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('../temp/train.csv')
    X = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    least_important_features, scores = RFE(X_train, y_train, X_test, y_test, 'gini', 2, columns=X.columns)
    print(least_important_features, scores)

    print(X.columns[least_important_features])