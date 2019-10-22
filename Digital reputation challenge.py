import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from  sklearn.decomposition import PCA
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")
os.chdir('./')
SEED = 17
params_XGboost = {'0': {'alpha': 0,
  'colsample_bytree': 0.6,
  'eta': 0.05,
  'learning_rate': 0.017,
  'max_delta_step': 5,
  'max_depth': 2,
  'min_child_samples': 100,
  'min_child_weight': 5,
  'n_estimators': 200,
  'num_leaves': 44,
  'scale_pos_weight': 1,
  'seed ': SEED},
 '1': {'alpha': 1,
  'colsample_bytree': 0.75,
  'eta': 0.225,
  'learning_rate': 0.033,
  'max_delta_step': 9,
  'max_depth': 1,
  'min_child_samples': 100,
  'min_child_weight': 2,
  'n_estimators': 200,
  'num_leaves': 19,
  'scale_pos_weight': 0.75,
  'seed ': SEED},
 '2': {'alpha': 0,
  'colsample_bytree': 0.75,
  'eta': 0.21,
  'learning_rate': 0.019,
  'max_delta_step': 3,
  'max_depth': 3,
  'min_child_samples': 150,
  'min_child_weight': 4,
  'n_estimators': 675,
  'num_leaves': 10,
  'scale_pos_weight': 0.75,
  'seed ': SEED},
 '3': {'alpha': 1,
  'colsample_bytree': 0.9,
  'eta': 0.39,
  'learning_rate': 0.021,
  'max_delta_step': 5,
  'max_depth': 1,
  'min_child_samples': 150,
  'min_child_weight': 7,
  'n_estimators': 325,
  'num_leaves': 20,
  'scale_pos_weight': 0.75,
  'seed ': SEED},
 '4': {'alpha': 1,
  'colsample_bytree': 0.7,
  'eta': 0.08,
  'learning_rate': 0.03,
  'max_delta_step': 9,
  'max_depth': 1,
  'min_child_samples': 50,
  'min_child_weight': 10,
  'n_estimators': 125,
  'num_leaves': 20,
  'scale_pos_weight': 0.75,
  'seed ': SEED }}

paramsCatBoost = {
'0': {'loss_function':'Logloss',
      'verbose': False,                        
      'random_seed': SEED,
      'thread_count': 1,
      'iterations': 130,
      'depth': 2,
      'border_count': 128     
     },    
'1': {'loss_function':'Logloss',
      'verbose': False,                        
      'random_seed': SEED,
      'thread_count': 1,
      'iterations': 90,
      'depth': 2,
      'border_count': 128     
     },
'2': {'loss_function':'Logloss',
      'verbose': False,                        
      'random_seed': SEED,
      'thread_count': 1,
      'iterations': 100,
      'depth': 2,
      'border_count': 128     
     },
'3': {'loss_function':'Logloss',
      'verbose': False,                        
      'random_seed': SEED,
      'thread_count': 1,
      'iterations': 90,
      'depth': 2 ,
      'border_count': 254    
     },          
'4': {'loss_function':'Logloss',
      'verbose': False,                        
      'random_seed': SEED,
      'thread_count': 1,
      'iterations': 70,
      'depth': 1,
      'border_count': 128     
     }}

def transform_data(data, n_components):
    rows, row_pos = np.unique(data.iloc[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data.iloc[:, 1], return_inverse=True)
    sparse_matrix = np.zeros((len(rows), len(cols)))
    sparse_matrix[row_pos, col_pos] = 1    
    scaler = StandardScaler(with_std=False)
    pca = PCA(n_components=n_components, random_state=SEED)
    cols_ = sparse_matrix.sum(axis=0)
    rows_ = sparse_matrix.sum(axis=1)
    minimum_users_per_group = 5
    selected_cols = cols_ >= minimum_users_per_group
    trimmed_sparse_matrix = sparse_matrix[:, selected_cols]
    transformed_data = pca.fit_transform(scaler.fit_transform(
        trimmed_sparse_matrix))
    return transformed_data

def train_and_predict(model, X_train, y_train, 
                            X_test, cv, scoring='roc_auc'):        
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                scoring=scoring)   
    print('CV mean: {}'.format(round(cv_scores.mean(), 5)))
    model.fit(X_train, y_train)    
    return model.predict_proba(X_test)[:, 1], cv_scores.mean()    

def write_to_submission_file(probas, ID, filename): 
    tmp = pd.DataFrame(probas).T
    baseline = pd.DataFrame(tmp.values, 
                            columns=['1', '2', '3', '4', '5'])
    baseline['id'] = ID
    baseline[['id', '1', '2', '3', '4', '5']].to_csv(filename, 
                                                     index=False)


#read data
X1_train = pd.read_csv('X1.csv')
X2_train = pd.read_csv('X2.csv')
X3_train = pd.read_csv('X3.csv')
X1_test = pd.read_csv('X1_test.csv')
X2_test = pd.read_csv('X2_test.csv')
X3_test = pd.read_csv('X3_test.csv')
target = pd.read_csv('Y.csv')
target = target.iloc[:, 1:]    

#plot corr matrix
for i in range(5):
    plt.figure(figsize=(15, 7))
    X1_train['target'] = target.iloc[: , i]
    sns.heatmap(np.abs(X1_train.drop('id', axis=1).corr()),
             linewidths=0.1, vmax=1.0, cmap='coolwarm')
    plt.suptitle(i)
    plt.show()

#plot count plot
sns.set_style('whitegrid')
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 5), sharey=True)
for i in range(5):
    sns.countplot(target.iloc[:, i], ax=axes[int(i / 2), i % 2],
                  palette='YlOrRd')


grouped_train = X2_train.groupby(['id'])['A'].agg('mean').reset_index()
grouped_test = X2_test.groupby(['id'])['A'].agg('mean').reset_index()

#use PCA for dimensionality reduction 
n_components = 2
scaler = StandardScaler(with_std=False)
pca = PCA(n_components=n_components, random_state=SEED)
full_data = X3_train.append(X3_test, sort=False).reset_index(drop=True)
transformed_data = pca.fit_transform(scaler.fit_transform(full_data))
X3_train_transformed = pd.DataFrame()
X3_test_transformed = pd.DataFrame()
X3_train_transformed['id'] = X3_train.id
X3_test_transformed['id'] = X3_test.id
for i in range(n_components):
    X3_train_transformed['pca' + str(i)] = transformed_data[:len(X3_train), i]
    X3_test_transformed['pca' + str(i)] = transformed_data[len(X3_train):, i]
del full_data


full_data = X2_train.append(X2_test, sort=False).reset_index(drop=True)
X2_train_transformed = pd.DataFrame()
X2_test_transformed = pd.DataFrame()
X2_train_transformed['id'] = X2_train.id.unique().tolist()
X2_test_transformed['id'] = X2_test.id.unique().tolist()

transformed_data = transform_data(full_data, n_components=n_components)
for i in range(n_components):
    X2_train_transformed['pca' + str(i)] = transformed_data[:len(X3_train), i]
    X2_test_transformed['pca' + str(i)] =  transformed_data[len(X3_train):, i]
del full_data

#merge dataframes
X_train = X1_train.merge(grouped_train, on='id', 
                how='inner').merge(X3_train_transformed,
                on='id', how='left').merge(
                X2_train_transformed, on='id', how='left')

X_test = X1_test.merge(grouped_test, on='id', 
                   how='inner').merge(X3_test_transformed,
                   on='id', how='left').merge(X2_test_transformed, 
                                              on='id', how='left')
del X1_train, X2_train, X3_train, X1_test, X2_test, X3_test

#drop useless columns
ID = X_test.id
for data in [X_train, X_test]:    
    data.drop('id', inplace=True, axis=1)
for col in X_train.columns:
    if X_train[col].nunique() == 1:
        X_train.drop(col, axis=1, inplace=True)     
        X_test.drop(col, axis=1, inplace=True)

#train models
models = {
    'xgboost': {'threshold': 0.0005, 'filename': 'predictions1.csv' },
    'catboost': {'threshold': 0.004, 'filename': 'predictions2.csv'}
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for key, value in models.items():
  cv_scores = []
  predictions = []
  for i in range(5):     
      if key == 'xgboost':
          model = XGBClassifier(**params_XGboost[str(i)])
      elif key == 'catboost':     
          model = CatBoostClassifier(**paramsCatBoost[str(i)])       

      y = target.iloc[:, i]              
      train_X, val_X, train_y, val_y = \
                   train_test_split(X_train, y, random_state=SEED, 
                                    shuffle=True)
      model.fit(train_X.values, train_y.values)
      perm = PermutationImportance(model, cv=5, 
                                    scoring='roc_auc', 
                                   random_state=SEED)
      perm.fit(val_X.values, val_y.values)
      sel = SelectFromModel(perm, threshold=value['threshold'], 
                            prefit=True)
      X_train_transformed = sel.transform(X_train)
      X_test_transformed = sel.transform(X_test)        
      
      prediction, cv_scores_mean = train_and_predict(model,
                        X_train_transformed, y.values,
                        X_test_transformed, cv)   
      
      cv_scores.append(cv_scores_mean)
      predictions.append(prediction)

  print(round(np.array(cv_scores).mean(), 5))  
  write_to_submission_file(predictions, ID, value['filename'])  


#simple blending
predictions1 = pd.read_csv('predictions1.csv')
predictions2 = pd.read_csv('predictions2.csv')
predictions = pd.DataFrame(predictions1.iloc[:, 1:].
                           add(predictions2.iloc[:, 1:]) * 0.5,
                           columns=['1', '2', '3', '4', '5'])
predictions['id'] = predictions1.id
predictions[['id', '1', '2', '3', '4', '5']].to_csv('predictions.csv', 
                                                    index=False)
