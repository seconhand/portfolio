import pandas as pd
import numpy as np
from optuna_rf import study_rfr
from optuna_rf import min_max_scaler 
from new_prep import value_prep_dt, basic_prep_dt, expand_prep_dt,power_prep_dt,clutch_prep_dt,batter_box_prep_dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#%%
data  = pd.read_excel('bat_data/21_23_value.xlsx')

data = data[~data['팀'].str.contains('P')]
#RPW :승리"당" 득점
# 각 지표 평균
data_df = data.iloc[:,3:]
for col in data_df.columns:
    if col == "RPW":
        continue
    data_df[col] = data_df[col] / 3
#%%
study = study_rfr(value_prep_dt)

print("best_value(mse) :", study.best_value,end="\n")
print("best_params : ", study.best_params)
'''
Best_Score :  0.015102435191653612
Best trial :  {'n_estimators': 4155,
               'max_depth': 10,
               'min_samples_split': 5,
               'min_samples_leaf': 3}
'''
#%%
dfs = [basic_prep_dt, expand_prep_dt,value_prep_dt,power_prep_dt,clutch_prep_dt
       ,batter_box_prep_dt]
for df in dfs:
    study = study_rfr(df)
    print("best_value(mse) :", study.best_value,end="\n")
    print("best_params : ", study.best_params)
    print('-'*30)
#%%    
'''
#BASIC
best_value(mse) : 0.10485658324601674
best_params :  {'n_estimators': 4155, 
                'max_depth': 10, 
                'min_samples_split': 5, 
                'min_samples_leaf': 3}
#EXPAND
best_value(mse) : 0.29849139280325676
best_params :  {'n_estimators': 365, 
                'max_depth': 6, 
                'min_samples_split': 9, 
                'min_samples_leaf': 4}
# VALUE
Best_Score :  0.015102435191653612
Best trial :  {'n_estimators': 4155,
               'max_depth': 10,
               'min_samples_split': 5,
               'min_samples_leaf': 3}
#POWER
best_value(mse) : 0.8838768587383306
best_params :  {'n_estimators': 4155, 
                'max_depth': 10, 
                'min_samples_split': 5, 
                'min_samples_leaf': 3}
#CLUTCH
best_value(mse) : 0.26112204971432207
best_params :  {'n_estimators': 1470,
                'max_depth': 4,
                'min_samples_split': 6,
                'min_samples_leaf': 7}
#BATTER_BOX
best_value(mse) : 1.4360713017397853
best_params :  {'n_estimators': 4777,
                'max_depth': 10,
                'min_samples_split': 4,
                'min_samples_leaf': 10}
'''
#%%
#21-23 타자 데이터
data  = pd.read_excel('bat_data/21_23_value.xlsx')

#투수의 타격기록 제외
data = data[~data['팀'].str.contains('P')]

# 각 지표 평균
data_df = data.iloc[:,3:]
for col in data_df.columns:
    #RPW :승리"당" 득점
    if col == "RPW":
        continue
    data_df[col] = data_df[col] / 3

# 예측데이터 Scaling
data_df_scale = min_max_scaler(data_df)
    
# 학습데이터 Scaling
X_data = min_max_scaler(value_prep_dt)
y_data = value_prep_dt['WAR*']

#HyperParameter 를 적용한 RandomForest 생성
model_rfr = RandomForestRegressor(**study.best_params)

#학습
model_rfr.fit(X_data,y_data)

#예측
pred = model_rfr.predict(data_df_scale)

pred_df = pd.DataFrame({"이름":data['이름'].values,"pred":pred})

pred_df.to_excel('24s_bat_predict_war.xlsx')
#%%
#==============================================================================
# DeepLearning 모델
import deepLearning
from tensorflow.keras.callbacks import EarlyStopping

df_dl = pd.read_excel('bat_data/DL_df.xlsx', index_col=0)
#%%
#scaling
df_dl = df_dl.iloc[:,2:]
scaler = MinMaxScaler()

target = df_dl['WAR*']
#%%
df_dl.drop(['WAR*','타석.1'],axis=1,inplace=True)
df_dl_scale = scaler.fit_transform(df_dl)

X_tr,X_val,y_tr,y_val = train_test_split(df_dl_scale,target,
                                         test_size=0.2)
#%%
model = deepLearning.MLP_model_creator(X_tr, y_tr)
model, history = deepLearning.MLP(model, X_tr, y_tr,X_val,y_val)
hist = history()

model.evaluate(X_val,y_val)
'''
[mse, mae]
[0.6987705826759338, 0.6996956467628479]
'''
#%%
# 손실함수
deepLearning.plot_loss_curve(hist,total_epoch=200)
deepLearning.plot_loss_curve(hist,total_epoch=50,start=1)

#%%
# earlystopping을 이용
model_2 = deepLearning.MLP_model_creator(X_tr, y_tr)
model_2, history_2 = deepLearning.MLP(model_2, X_tr, y_tr, X_val, y_val)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
hist_2 = history_2(callbacks=early_stopping)
#%%
model_2.evaluate(X_val,y_val)
#%%
# 손실함수
deepLearning.plot_loss_curve(hist_2,total_epoch=38)
y_pred_2 = model_2.predict(X_val)
'''
[mse, mae]
[0.640755295753479, 0.6503634452819824]
'''
#%%============================================================================
basic_2123 = pd.read_excel('bat_data/21_23_basic.xlsx')
expand_2123 = pd.read_excel('bat_data/21_23_expand.xlsx')
value_2123 = pd.read_excel('bat_data/21_23_value.xlsx')
power_2123 = pd.read_excel('bat_data/21_23_power.xlsx')
clutch_2123 = pd.read_excel('bat_data/21_23_clutch.xlsx')
batter_box_2123 = pd.read_excel('bat_data/21_23_batter_box.xlsx')

dfs = [basic_2123, expand_2123, value_2123,power_2123,clutch_2123,batter_box_2123]

expand_2123.columns = [x[:-2] +'_파크팩터'if '.1' in x else x for x in expand_2123.columns]
clutch_2123.columns = [x[:-2]+'_WPA' if '.1' in x else x for x in clutch_2123.columns]

for df in dfs:
    df.drop('순',axis= 1,inplace =True)

df_2123 = pd.merge(basic_2123,expand_2123,on=['이름','팀'])
#중복된 컬럼 삭제
df_2123.drop(df_2123.columns[df_2123.columns.str.contains('_y')],axis=1, inplace = True)
df_2123.columns = df_2123.columns.map(lambda x : x[:-2] if "_x" in x else x)
   
df_2123 = pd.merge(df_2123,value_2123,on=['이름','팀'])
df_2123.drop(df_2123.columns[df_2123.columns.str.contains('_y')],axis=1, inplace = True)
df_2123.columns = df_2123.columns.map(lambda x : x[:-2] if "_x" in x else x)   

df_2123 = pd.merge(df_2123,power_2123,on=['이름','팀'])
df_2123.drop(df_2123.columns[df_2123.columns.str.contains('_y')],axis=1, inplace = True)
df_2123.columns = df_2123.columns.map(lambda x : x[:-2] if "_x" in x else x)   

df_2123 = pd.merge(df_2123,clutch_2123,on=['이름','팀'])
df_2123.drop(df_2123.columns[df_2123.columns.str.contains('_y')],axis=1, inplace = True)
df_2123.columns = df_2123.columns.map(lambda x : x[:-2] if "_x" in x else x)   

df_2123 = pd.merge(df_2123,batter_box_2123,on=['이름','팀'])
df_2123.drop(df_2123.columns[df_2123.columns.str.contains('_y')],axis=1, inplace = True)
df_2123.columns = df_2123.columns.map(lambda x : x[:-2] if "_x" in x else x)   
#%%
# 3년치 평균
names = df_2123['이름']
df_2123 = df_2123[df_dl.columns]

for col in df_2123.columns:
    if col == "RPW":
        continue
    df_2123[col] = df_2123[col] / 3
#%%
#scaling
scaler = MinMaxScaler()
df_2123_scale = scaler.fit_transform(df_2123)
#%%
pred1 = model.predict(df_2123_scale)
pred2 = model_2.predict(df_2123_scale)
#%%
print(pred1)
print(pred2)
#%%
pred_df = pd.DataFrame({"이름":names.values,
                         "nostop_perd":np.squeeze(pred1),
                         "stop_pred":np.squeeze(pred2)})
#%%
pred_df.to_excel('24s_bat_DL_predict_war.xlsx')






