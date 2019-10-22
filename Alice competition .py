import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import eli5
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import re
from scipy.stats.mstats import gmean
import warnings
warnings.filterwarnings("ignore")
os.chdir('./')
SEED = 17
time_split = TimeSeriesSplit(n_splits=10)
skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=SEED)
logit = LogisticRegression(C=1, random_state=SEED, solver='liblinear')

#add time features
def add_time_features(train_times, test_times, X_sparse_train, X_sparse_test,
                                                  scaler, n_sites_train, n_sites_test): 
          
    times = ['time%s' % i for i in range(1, 11)]
    train_times_len = len(train_times)
    full_times = pd.concat([train_times, test_times])    
    time_df = pd.DataFrame(index=full_times.index)   
    
    month = full_times['time1'].apply(lambda x: x.month)
    day = full_times['time1'].apply(lambda x: x.day)   
    hour = full_times['time1'].apply(lambda x: x.hour)
    dayofweek = full_times['time1'].apply(lambda x: x.dayofweek)
    min_val = full_times[times].min(axis=1)
    max_val = full_times[times].max(axis=1)

    
    
    time_df['month_day_11_12'] = ((month == 11) & (day == 12)).astype('int')    
    
    
    time_df['year'] = full_times['time1'].apply(lambda x: x.year)
    time_df['duration'] = ((max_val - min_val) / np.timedelta64(1, 's')) / 1800
    
    time_df['day_3'] = (day == 3).astype('int')
    time_df['day_21'] = (day == 21).astype('int')
    time_df['day_26'] = (day == 26).astype('int')    
    time_df['day_23'] = (day == 23).astype('int')
    
    
    time_df['month_10'] = (month == 10).astype('int')     
    
    time_df['isWednesday'] = (dayofweek == 2).astype('int')
    time_df['isSunday'] = (dayofweek == 6).astype('int')   
    time_df['19-08h'] = (((hour >= 19) & (hour <= 23)) | ((hour >= 0) & (hour <= 8))).astype('int')
    time_df['10-11h'] = ((hour >= 10) & (hour <= 11)).astype('int')
    time_df['14h'] = ((hour >= 14) & (hour <= 14)).astype('int')
    time_df['month12_day_greater_18'] = ((month == 12) & (day >= 18)).astype('int')
    
    
    time_df['month4_day_greater_15'] = ((month == 4) & (day > 15)).astype('int')
    col_for_dummies = ['year']  
    time_df_dummies = pd.get_dummies(time_df, columns=col_for_dummies)
    
    
    X_sparse_train_full = hstack([X_sparse_train, time_df_dummies[:train_times_len]])
    X_sparse_test_full = hstack([X_sparse_test, time_df_dummies[train_times_len:]])
        
    return X_sparse_train_full, X_sparse_test_full, list(time_df_dummies.columns) 

def train_and_predict(model, X_train, y_train, X_test, site_feature_names=vectorizer.get_feature_names(), 
                      new_feature_names=None, cv=time_split, scoring='roc_auc',
                      top_n_features_to_show=30):
   
    cv_scores1 = cross_val_score(model, X_train, y_train, cv=cv, 
                            scoring=scoring, n_jobs=-1)
    cv_scores2 = cross_val_score(model, X_train, y_train, cv=skf, 
                            scoring=scoring, n_jobs=-1)    
    
    print('CV scores time split', cv_scores1)
    print('CV scores skf', cv_scores2)
    print('')
    print('CV mean 1: {}, CV std 1: {}'.format(round(cv_scores1.mean(), 5), round(cv_scores1.std(), 5)))
    print('CV mean 2: {}, CV std 2: {}'.format(round(cv_scores2.mean(), 5), round(cv_scores2.std(), 5)))
    
    print('gmean: {}'.format(gmean([cv_scores1.mean(), cv_scores2.mean()])))    
    
    model.fit(X_train, y_train)
    
    if new_feature_names:
        all_feature_names = site_feature_names + new_feature_names 
    else: 
        all_feature_names = site_feature_names
    
    #display_html(eli5.show_weights(estimator=model, 
    #              feature_names=all_feature_names, top=top_n_features_to_show))
    
    if new_feature_names:
        print('New feature weights:')
        df1 = pd.DataFrame({'feature': new_feature_names, 
                        'coef': model.coef_.flatten()[-len(new_feature_names):]}).sort_values(by='coef', 
              ascending=False) 
    
        print(df1)
    
    test_pred = model.predict_proba(X_test)[:, 1]
    #write_to_submission_file(test_pred) 
    
    return cv_scores1

def prepare_sparse_features():
    print('Start...')    
    scaler = MinMaxScaler()
    times = ['time%s' % i for i in range(1, 11)]
    sites = ['site%s' % i for i in range(1, 11)]
    
    train_df = pd.read_csv('train_sessions.csv',
                           index_col='session_id', parse_dates=times)
    test_df = pd.read_csv('test_sessions.csv',
                          index_col='session_id', parse_dates=times)           
    
    train_df = train_df.sort_values(by='time1')
    y = train_df.target.values
    with open('site_dic.pkl', 'rb') as f:
        site2id = pickle.load(f)

    id2site = { v:k for (k, v) in site2id.items() }
    id2site[0] = 'unknown'
    for key, value in id2site.items():
        id2site[key] = re.sub("^\S*?\.*?www\S*?\.", '', value) 
        
    sites = ['site%s' % i for i in range(1, 11)]    

    #train
    train_sessions = train_df[sites].fillna(0).astype(np.int32).apply(lambda row:
                                                    ' '.join([id2site[i] for i in row]), axis=1)    
    train_n_unique_sites = train_sessions.apply(lambda x: 
                                len(np.unique(x.replace('unknown', '').split()))).values.reshape(-1, 1) 
    
    train_n_sites = train_sessions.apply(lambda x: 
                                len(x.replace('unknown', '').split())).values.reshape(-1, 1)  
    
    train_n_sites_scaled = train_n_sites / 10      

    #test    
    test_sessions = test_df[sites].fillna(0).astype(np.int32).apply(lambda row: 
                                                    ' '.join([id2site[i] for i in row]), axis=1)    
    
    test_n_sites = test_sessions.apply(lambda x: 
                            len(x.replace('unknown', '').split())).values.reshape(-1, 1)    
    
    test_n_sites_scaled = test_n_sites / 10    
   
    vectorizer = TfidfVectorizer(max_features=50000, 
                                 ngram_range=(1, 5), tokenizer=lambda x: x.split())

    sparse_train = vectorizer.fit_transform(train_sessions)
    sparse_test = vectorizer.transform(test_sessions)    
    
    full_train = hstack([sparse_train]).tocsr()
    full_test = hstack([sparse_test]).tocsr()
    
    train_times, test_times = train_df[times], test_df[times]    
    
    return full_train, full_test, y, train_times, test_times, vectorizer, train_n_sites, test_n_sites     




#train model
sparse_train, sparse_test, y, train_times, test_times, vectorizer, train_n_sites, test_n_sites \
                                            = prepare_sparse_features()
scaler = MinMaxScaler()
sparse_train, sparse_test, feature_names = add_time_features(train_times, test_times,
                                            sparse_train, sparse_test, scaler, train_n_sites, test_n_sites)

cv_scores = train_and_predict(logit, sparse_train, y, sparse_test, 
                             site_feature_names=vectorizer.get_feature_names(),
                             new_feature_names=feature_names)    