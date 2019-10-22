#https://www.kaggle.com/c/ieee-fraud-detection
# Private leaderboard - 0.921864

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn import metrics, preprocessing
import lightgbm as lgbm
import seaborn as sns
import datetime
import warnings
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
os.chdir('./')
SEED = 17
time_split = TimeSeriesSplit(n_splits=5)
params = {'learning_rate': 0.03,
          'objective': 'binary',
          'metric': 'auc',
          'num_threads': 1,
          'num_leaves': 256,
          'verbose': 1,
          'random_state': SEED, 
          'device' : 'gpu',
          'gpu_platform_id': 0,
          'gpu_device_id':0         
         }
lightgbm = lgbm.LGBMClassifier(**params,  n_estimators=270)
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

def train_and_predict(model, X_train, y_train, X_test,
                      cv, transactionID, scoring='roc_auc'):    
    print('Start cross validation...')
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                scoring=scoring)
    print('CV scores time split', cv_scores)
    print('CV mean: {}'.format(round(cv_scores.mean(), 5)))
    model.fit(X_train, y_train)    
    test_predictions = model.predict_proba(X_test)[:, 1]
    write_to_submission_file(transactionID, test_predictions)         

def write_to_submission_file(transactionID, y_test):    
    data = {'TransactionID':transactionID, 'isFraud':y_test}
    submissionDF = pd.DataFrame(data)
    submissionDF.to_csv('sample_submission.csv', index=False)       

def select_from_model_lgb(X, y):
    print('select_from_model_lgb')    
    embeded_lgb_selector = SelectFromModel(lightgbm, threshold='median')
    embeded_lgb_selector.fit(X, y)        
    return embeded_lgb_selector.get_support() 
 


 #read data
train_identity_data = pd.read_csv('train_identity.csv')
train_transaction_data = pd.read_csv('train_transaction.csv')
test_identity_data = pd.read_csv('test_identity.csv')
test_transaction_data = pd.read_csv('test_transaction.csv')

target =  train_transaction_data.isFraud.values
transactionID = test_transaction_data.TransactionID
X_train = pd.merge(train_transaction_data, train_identity_data, how='left')
X_test = pd.merge(test_transaction_data, test_identity_data, how='left')

del train_identity_data, train_transaction_data, \
                            test_identity_data, test_transaction_data

col_to_drop = ['V300','V309','V111','V124','V106','V125',
              'V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286',
              'V318','V304','V116','V284','V293',
              'V137','V295','V301','V104','V311','V115',
              'V109','V119','V321','V114','V133','V122',
              'V319','V105','V112','V118','V117','V121','V108',
              'V135','V320','V303','V297','V120',
              'V1','V14','V41','V65','V88', 'V89',
              'V107', 'V68', 'V28', 'V27', 'V29', 
              'V241','V269', 'V240', 'V325', 'V138',
              'V154', 'V153', 'V330', 'V142', 'V195',
              'V302', 'V328', 'V327','V198', 'V196', 'V155']

new_browsers =['samsung browser 7.0', 
               'opera 53.0',
               'mobile safari 10.0', 
               'google search application 49.0',
               'firefox 60.0', 
               'edge 17.0', 
               'chrome 69.0', 
               'chrome 67.0',
               'chrome 63.0', 
               'chrome 63.0', 
               'chrome 64.0', 
               'chrome 64.0 for android', 
               'chrome 64.0 for ios', 
               'chrome 65.0', 
               'chrome 65.0 for android',
               'chrome 65.0 for ios', 
               'chrome 66.0', 
               'chrome 66.0 for android', 
               'chrome 66.0 for ios']

X_train.drop(col_to_drop, axis=1, inplace=True)              
X_test.drop(col_to_drop, axis=1, inplace=True)

C_col = [f'C{i}' for i in range(1, 15)]
M_col = [f'M{i}' for i in range(1, 10)]
D_col = [f'D{i}' for i in range(1, 16)]
D_col_except_D9 = D_col.copy()
del D_col_except_D9[8]

card_col = [f'card{i}' for i in range(1, 7)]
col_to_drop = ['TransactionID', 'TransactionDT', 'date']

#create features
for data in [X_train, X_test]:
      data['date'] = data['TransactionDT'].apply(lambda x:
                          (startdate + datetime.timedelta(seconds=x)))
      data['hour'] = data['date'].dt.hour.astype('str')   
      data['day'] = data['date'].dt.day.astype('str')
      data['month'] = data['date'].dt.month.astype('str')
      data['weekday'] = data['date'].dt.dayofweek.astype('str')
      data['weekday__hour'] = data['weekday'] + '_' + data['hour']
      data['month__day'] = data['month'] + '_' + data['day']
      data['P_emaildomain_R_emaildomain'] = \
          (data['P_emaildomain'] + '_' + data['R_emaildomain']).astype('object')
      data['null_count'] = data.isnull().sum(axis=1)    
      data['TransactionAmt_frac'] = \
                      data['TransactionAmt'] - np.fix(data['TransactionAmt'])
      data['TransactionAmt_decimal_lenght'] = \
            data['TransactionAmt'].astype(str).str.split('.',
                                                expand=True)[1].str.len()
      data['addr1_addr2_null'] = (data['addr1'].isnull() 
                              & X_train['addr2'].isnull()).astype('int')                                              
      data['dist1_dist2_null'] = (data['dist1'].isnull() 
                              & X_train['dist2'].isnull()).astype('int')                                                                          
      data['sum_Ci'] = data[C_col].sum(axis=1)                        
      data['null_M'] = data[M_col].isnull().sum(axis=1)    
      data['null_D'] = data[D_col].isnull().sum(axis=1)                       
      data['D9_null'] = data.D9.isnull().astype('int')
      data['id_31_null'] = data.id_31.isnull().astype('int')    
      data['email_addr_are_equal'] = \
                  (data.R_emaildomain == data.P_emaildomain).astype('int')             
      data['card_1_dist1'] = data.card1.astype('str') + '_' + \
                data.dist1.astype('str')
      data['card1_P_emaildomain'] = data.card1.astype('str') + '_' + \
                data.P_emaildomain.astype('str')
      data['addr1_card1'] = data.addr1.astype('str') + '_' + \
                data.card1.astype('str')
      data['card4_dist1'] = data.card4.astype('str') + '_' + \
                data.dist1.astype('str')
      data['addr1_card4'] = data.addr1.astype('str') + '_' + \
                data.card4.astype('str') 

      data['P_emaildomain_C2'] = data.P_emaildomain.astype('str') + '_' + \
                data.C2.astype('str') 
      data['addr1_card2'] = data.addr1.astype('str') + '_' + \
                data.card2.astype('str')
      data['card5_P_emaildomain'] = data.card5.astype('str') + '_' + \
                data.P_emaildomain.astype('str') 
      data['card1_card5'] = data.card1.astype('str') + '_' + \
                data.card5.astype('str')  
      data['new_browsers'] = data.id_31.isin(new_browsers).astype('int')

      for col in D_col_except_D9:
          data[col].fillna(data[col].median(), inplace=True)  
          data[col] = data[col].astype('int')
      data['UniqueDates'] = data[D_col_except_D9].nunique(axis=1)
      data['sumD'] = data[D_col_except_D9].sum(axis=1)
      data['DayOfYear'] = data['date'].dt.dayofyear.astype('str')   
      data['device_version'] = data['DeviceInfo'].str.split('/', expand=True)[1]      

      data.drop(col_to_drop, axis=1, inplace=True)

all_data = X_train.append(X_test, sort=False).reset_index(drop=True)
i_cols = ['card2', 'card3', 'card5', 'addr1', 'card4', 'addr2', 'ProductCD']
for col in i_cols:
    grouped = all_data.groupby(col) \
      ['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_' 
                                          + col + '_mean'}, axis=1)  
    all_data = pd.merge(all_data, grouped, on=col, how='left')
    X_train['TransactionAmt_' + col + '_mean'] = \
           list(all_data['TransactionAmt_' + col + '_mean'])[:len(X_train)]
    X_test['TransactionAmt_' + col + '_mean'] = \
      list(all_data['TransactionAmt_' + col + '_mean'])[len(X_train):]        
del all_data  

X_train.drop('isFraud', axis=1, inplace=True)

cat_columns = []
num_columns = []
for i in X_train.columns.tolist():
    if X_train[i].dtype=='object':
        cat_columns.append(i)
    else:
        num_columns.append(i)       

cat_data = X_train[cat_columns].append(X_test[cat_columns],
                                       sort=False).reset_index(drop=True)
for col in cat_columns:
    cat_data[col].fillna('-', inplace=True)    
    encoding = cat_data.groupby(col).size()
    encoding = encoding / len(cat_data)
    cat_data[col] = cat_data[col].map(encoding)
    X_train[col] = list(cat_data[col][:len(X_train)])
    X_test[col] = list(cat_data[col][len(X_train):].copy())
del cat_data

imputer = Imputer(strategy='median')
X_train[num_columns] = imputer.fit_transform(X_train[num_columns])
X_test[num_columns] =  imputer.transform(X_test[num_columns])

#feature selection
feature_names = X_train.columns.tolist()
light_gbm_support = select_from_model_lgb(X_train, target)
feature_selection_df = pd.DataFrame({'feature':feature_names, 
                                     'light_gbm':light_gbm_support                                                                                                             
                                    })

feature_selection_df['total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df = \
          feature_selection_df.sort_values(['total'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df) + 1)
col_to_drop = \
      feature_selection_df[feature_selection_df.total==0].feature.tolist()

X_train.drop(col_to_drop, axis=1, inplace=True)
X_test.drop(col_to_drop, axis=1, inplace=True)

X_train = X_train.values
X_test = X_test.values         

#modeling
train_and_predict(lightgbm, X_train, target, 
                           X_test, time_split, transactionID)
