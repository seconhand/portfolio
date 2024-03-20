import pandas as pd
import numpy as np
# Target = "WAR*"
# In[basic]
basic_dt = pd.read_excel('bat_data/basic.xlsx')
basic_dt = basic_dt.drop('순',axis = 1)
basic_prep_dt = basic_dt.dropna(axis = 0).reset_index(drop=True)

basic_prep_dt = basic_prep_dt.loc[:,"G":]
# war* 이 없는 데이터 결합(merge)시킬 DF
basic_war = basic_dt[['이름','팀','WAR*']]
# In[expandation]
expand_dt = pd.read_excel('bat_data/expandation.xlsx')
expand_dt.columns =[x[:-2] +'_파크팩터'if '.1' in x else x for x in expand_dt.columns]
expand_dt = expand_dt.drop('순',axis = 1)
# 병합
expand_dt = pd.merge(expand_dt,basic_war)

expand_prep_dt = expand_dt.dropna(axis = 0)
expand_prep_dt = expand_prep_dt.loc[:,"타석":].reset_index(drop=True)
# In[value]
value_dt = pd.read_excel('bat_data/value.xlsx')
value_dt = value_dt.drop('순',axis = 1)

value_dt = pd.merge(value_dt,basic_war)

value_prep_dt = value_dt[value_dt.타석 !=0]
value_prep_dt = value_prep_dt.dropna(axis=0)
value_prep_dt = value_prep_dt.loc[:,"타석":].reset_index(drop=True)
# In[clutch]
clutch_dt = pd.read_excel('bat_data/clutch.xlsx')
clutch_dt = clutch_dt.drop('순',axis = 1)
clutch_dt.columns = [x[:-2]+"_WPA" if '.1' in x else x for x in clutch_dt.columns]

clutch_dt = pd.merge(clutch_dt,basic_war,on =['이름','팀'])

clutch_prep_dt = clutch_dt[clutch_dt.타석 !=0]
clutch_prep_dt = clutch_prep_dt.dropna(axis=0)
clutch_prep_dt = clutch_prep_dt.loc[:,"타석":].reset_index(drop=True)
# In[batter_box]
batter_box_dt = pd.read_excel('bat_data/batter_box.xlsx')
batter_box_dt = batter_box_dt.drop('순',axis = 1)

batter_box_dt = pd.merge(batter_box_dt,basic_war)

batter_box_prep_dt = batter_box_dt[batter_box_dt.타석 !=0]
batter_box_prep_dt = batter_box_prep_dt.dropna(axis=0)
batter_box_prep_dt = batter_box_prep_dt.loc[:,"타석":].reset_index(drop=True)
# In[power]
power_dt = pd.read_excel('bat_data/power.xlsx')
power_dt = power_dt.drop('순',axis = 1)

power_dt = pd.merge(power_dt,basic_war)

power_prep_dt = power_dt[power_dt.타석 !=0]
power_prep_dt = power_prep_dt.dropna(axis=0)
power_prep_dt = power_prep_dt.loc[:,"타석":].reset_index(drop=True)
power_prep_dt = power_prep_dt.loc[power_prep_dt['PA/HR'] != 10000].reset_index(drop=True)
#%%
# q1,q3 = power_prep_dt['PA/HR'].quantile([0.25,0.75])
# IOQ = q3 - q1
# power_prep_dt = power_prep_dt[(power_prep_dt['PA/HR'] >= q1-1.5*IOQ) & 
#                              (power_prep_dt['PA/HR'] <= q3+1.5*IOQ)]
import opt_tuna

