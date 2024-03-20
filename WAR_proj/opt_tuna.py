import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
from optuna import Trial
from sklearn.metrics import mean_squared_error

import new_prep
#%%
def rf_objective(trial):
    params = {
    'n_estimators': trial.suggest_int('n_estimators',0, 5000),
    'criterion': 'squared_error',
    'max_depth' : trial.suggest_int('max_depth', 3, 10),
    'min_samples_split': trial.suggest_int('min_samples_split', 3, 10),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),
    'min_weight_fraction_leaf': 0.0,
    'max_features': 1.0,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'n_jobs': None,
    'random_state': 28
    }
    
    X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)
    
    model = RandomForestRegressor(**params)
    model.fit(X_tr,y_tr)
    
    y_pred = model.predict(X_val)
    mse  =  mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    return rmse
# In[scaler]
def st_scaler(x):
    scaler = StandardScaler()
    x = x.drop("WAR*",axis = 1)
    x_scale = scaler.fit_transform(x)
    
    return x_scale
# In[basic]
# new_prep 파일로 부터 데이터얻기
x = basic_prep_dt
y = x['WAR*']

# scaling
x = st_scaler(x)

#최적화학습
sampler = TPESampler(seed=28)
study = optuna.create_study(
    study_name="RFR_parameter_opt",
    direction= "minimize",
    sampler=sampler
    )
study.optimize(rf_objective,n_trials=10)
#%%
print("basic_Best score : ", study.best_value)
print("basic_Best trial : ", study.best_params)
#%%
# 최적 파라미터로 모델 학습 
basic_params = study.best_params
model_rf = RandomForestRegressor(**basic_params)

X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)

model_rf.fit(X_tr,y_tr)
y_pred = model_rf.predict(X_val)
print('basic_rmse : ', np.sqrt(mean_squared_error(y_val, y_pred)))
# In[expandation]
# new_prep 파일로 부터 데이터얻기
x = expand_prep_dt
y = x['WAR*']
#%%
# scaling
x = st_scaler(x)

#최적화학습
sampler = TPESampler(seed=28)
study = optuna.create_study(
    study_name="RFR_parameter_opt",
    direction= "minimize",
    sampler=sampler
    )
study.optimize(rf_objective,n_trials=10)
#%%
print("expandation_Best score : ", study.best_value)
print("expandation_Best trial : ", study.best_params)
#%%
expandation_params = study.best_params
model_rf = RandomForestRegressor(**basic_params)

X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)

model_rf.fit(X_tr,y_tr)
y_pred = model_rf.predict(X_val)
print('expand_rmse : ', np.sqrt(mean_squared_error(y_val, y_pred)))
# In[value]
# new_prep 파일로 부터 데이터얻기
x = value_prep_dt
y = x['WAR*']

# scaling
x = st_scaler(x)

#최적화학습
sampler = TPESampler(seed=28)
study = optuna.create_study(
    study_name="RFR_parameter_opt",
    direction= "minimize",
    sampler=sampler
    )
study.optimize(rf_objective,n_trials=10)
#%%
print("value_Best score : ", study.best_value)
print("value_Best trial : ", study.best_params)
#%%
value_params = study.best_params
model_rf = RandomForestRegressor(**basic_params)

X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)

model_rf.fit(X_tr,y_tr)
y_pred = model_rf.predict(X_val)
print('value_rmse : ', np.sqrt(mean_squared_error(y_val, y_pred)))
# In[batter_box]
# new_prep 파일로 부터 데이터얻기
x = batter_box_prep_dt
y = x['WAR*']

# scaling
x = st_scaler(x)

#최적화학습
sampler = TPESampler(seed=28)
study = optuna.create_study(
    study_name="RFR_parameter_opt",
    direction= "minimize",
    sampler=sampler
    )
study.optimize(rf_objective,n_trials=10)
#%%
print("batter_box_Best score : ", study.best_value)
print("batter_box_Best trial : ", study.best_params)
#%%
batter_box_params = study.best_params
model_rf = RandomForestRegressor(**basic_params)

X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)

model_rf.fit(X_tr,y_tr)
y_pred = model_rf.predict(X_val)
print('batter_box_rmse : ', np.sqrt(mean_squared_error(y_val, y_pred)))
# In[power]
# new_prep 파일로 부터 데이터얻기
x = power_prep_dt
y = x['WAR*']

# scaling
x = st_scaler(x)

#최적화학습
sampler = TPESampler(seed=28)
study = optuna.create_study(
    study_name="RFR_parameter_opt",
    direction= "minimize",
    sampler=sampler
    )
study.optimize(rf_objective,n_trials=10)
#%%
print("power_Best score : ", study.best_value)
print("power_Best trial : ", study.best_params)
#%%
power_params = study.best_params
model_rf = RandomForestRegressor(**basic_params)

X_tr,X_val,y_tr,y_val =train_test_split(x,y,test_size=0.2)

model_rf.fit(X_tr,y_tr)
y_pred = model_rf.predict(X_val)
print('power_rmse : ', np.sqrt(mean_squared_error(y_val, y_pred)))