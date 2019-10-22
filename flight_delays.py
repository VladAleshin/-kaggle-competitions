import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")
os.chdir('./') 
SEED = 17
TRAIN_LEN = 10**5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def read_prepare_data():
    '''
    clean data and simple FE
    '''
    train_df = pd.read_csv('flight_delays_train.csv')
    test_df = pd.read_csv('flight_delays_test.csv')
    y = train_df.dep_delayed_15min.map({'Y': 1, 'N': 0}).values
    train_df.drop('dep_delayed_15min', axis=1, inplace=True)
    full_df = pd.concat([train_df, test_df])
    
    full_df.Month = full_df.Month.apply(lambda s: s.replace('c-', '')).astype('int')    
    full_df.DayofMonth = full_df.DayofMonth.apply(lambda s: s.replace('c-', '')).astype('int')
    full_df.DayOfWeek = full_df.DayOfWeek.apply(lambda s: s.replace('c-', '')).astype('int')        
    full_df.DepTime = full_df.DepTime.astype('int') // 100        
    full_df.Distance = full_df.Distance.astype('int')  
    full_df.DepTime[full_df.DepTime == 25] = 1    
    full_df.DepTime[full_df.DepTime == 24] = 0               
    cat_features = ['Month', 
                    'DayofMonth', 
                    'DayOfWeek', 
                    'DepTime', 
                    'UniqueCarrier',
                    'Origin', 
                    'Dest'
                   ]
    return full_df, y, cat_features  

#modeling
full_df, y, cat_features = read_prepare_data()
full_df.reset_index(drop=True, inplace=True)
train, test = full_df[:TRAIN_LEN], full_df[TRAIN_LEN:]
cat_model = CatBoostClassifier(loss_function='Logloss',
                               verbose=False,
                               cat_features=cat_features, 
                               random_seed=SEED,
                               thread_count=4)
params = {'iterations': [250, 260, 270, 280, 290, 300],
          'depth': [6, 7, 8, 9, 10]          
         }
clf = GridSearchCV(cat_model, param_grid=params, 
                         cv=skf, scoring='roc_auc', n_jobs=-1, verbose=1)
clf.fit(train, y)
clf.best_estimator_.fit(train, y)
predictions = clf.best_estimator_.predict_proba(test)[:, 1]    