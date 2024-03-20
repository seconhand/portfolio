# 각 지표별 WAR 과의 상관관계 분석 및 시각화
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from new_prep import basic_prep_dt, expand_prep_dt, value_prep_dt,power_prep_dt,\
    clutch_prep_dt,batter_box_prep_dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
    
# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

#%%

dfs = [basic_prep_dt, expand_prep_dt, value_prep_dt,power_prep_dt,\
    clutch_prep_dt,batter_box_prep_dt]
#%% 시각화 함수
def visualization(D):
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False
    plt.rc('axes', labelsize=20)
    
    l = int((len(D.columns)-1)/6)
    m = 0
    
    for n in range(l+1):
        plt.figure(figsize = (20,20))
        for idx, col in enumerate(D.columns[1+m:7+m]):
    
            ax1 = plt.subplot(3,2, idx+1)
            sns.regplot(x = col, y= 'WAR*', 
                        line_kws={'color': 'red' },
                        data = D,ax = ax1)
        m += 6    
        plt.show()       
#%% 정규성 검정
# 정규성 판정
# 일반적으로 데이터양이 많으면 kstest로 실행
import  scipy.stats as stats

dts = []
for data in dfs:
    cols = data.columns
    norm = []
    norm_x = []
    for col in cols:
        statics, pvalue = stats.kstest(data[col], 'norm')
        # 유의확률 5%에서 검정
        if pvalue < 0.05:
            print(f'{col}은 정규분포를 따르지 않는다.')
            print(f'통계량 : {statics}, p-val:{pvalue}')
            norm_x.append(col)
            
        else:
            print(f'{col}은 정규분포를 따른다.')
            print(f'통계량 : {statics}, p-val:{pvalue}')
            norm.append(col)
    print("norm_x",norm_x)        
    print("norm",norm)

    data = data[norm_x]
    dts.append(data)            
    print()    
# 전부 정규성을 만족하지 않는다.
#%% SPEARMAN 상관분석 및 시각화
# 상관관계분석
# 정규성을 만족하지 않으면 spearman 상관관계 분석 실행
# 내림차순으로 정렬
# WPR* 상관관계 분석

f_names = ["basic", 'expaned', 'value', 'power','clutch' ,'batter']
for idx,dt in enumerate(dts):
    statics = dt.corr(method = 'spearman')['WAR*'].sort_values(ascending = False)
    print(statics.head(15))
    
    #상관계수 엑셀
    statics_df = statics.sort_values(ascending=False)
    statics_df = pd.DataFrame(statics_df).rename(columns={'WAR*':'corr'}).iloc[1:,:]
    statics_df.to_excel(f'{f_names[idx]}_corr.xlsx')
    

    #scaling
    rel_cols = statics.index
    plot_df = dt[rel_cols]
    
    scaler = MinMaxScaler()
    plot_df_scale = scaler.fit_transform(plot_df.iloc[:,1:])
    plot_df_scale = pd.DataFrame(data= plot_df_scale,columns=rel_cols[1:])
    plot_df_scale = pd.concat([plot_df['WAR*'],plot_df_scale],axis=1)    
    
    #시각화
    visualization(plot_df_scale)   
    
#%% 상관계수 결과 상위 15개만 표시

'''
1. BASIC     
wRC+    0.829165
wOBA    0.821311
OPS     0.812893
출루      0.771887
장타      0.764837
타율      0.744000
루타      0.728510
타점      0.718938
안타      0.718452
2타      0.717633
득점      0.710493
볼넷      0.708772
홈런      0.705493
타석      0.673014

2. EXPAND
wRAA_파크팩터      0.876811
wRAA           0.871470
wRC+           0.864448
wRC_파크팩터       0.851434
wRC            0.845668
wOBA_파크팩터      0.836522
wRC/27_파크팩터    0.831897
wOBA           0.831641
wRC/27         0.826744
IsoP           0.526777
타석             0.480132
BABIP          0.438003
HR%            0.421596
BB%            0.369847

3. VALUE
WAROff     0.961136
RAR        0.958847
WAR        0.958789
WARBat     0.956177
타석         0.872557
대체Run      0.868488
공격_RAA     0.679432
타격         0.656506
RAA-Adj    0.644749
WAA        0.611248
RAA        0.609675
RPW        0.104172
주루         0.064958
도루         0.048206

4. POWER
장타       0.641462
XH/AB    0.577715
IsoP     0.514592
타석       0.478679
홈런       0.467950
1점       0.447600
2점       0.409220
중        0.399768
XH/H     0.397764
HR/외뜬    0.380478
3점       0.334467
우        0.296403
HR/XH    0.263555
좌        0.229188

5. CLUTCH
Plus      0.757348
타점        0.753722
순수        0.746273
WPA/LI    0.728739
PA        0.716601
타석        0.713828
주자        0.710510
RE24      0.653360
REW       0.650905
타격        0.631377
WPA       0.631354
타격_WPA    0.606790
순수/주자수    0.558011
득점권타율     0.527721

6. BATTER_BOX
타석        0.480132
투구        0.459554
스윙.1      0.458749
B         0.426269
N         0.338871
2S후선구%    0.304804
W%        0.249438
스윙        0.076249
P/PA      0.073416
헛스윙       0.048022
루킹.1      0.036025
파울        0.029949
타격        0.026516
초구        0.020360
'''   
#%%============================================================================
# 상관계수가 0.5 이상이 컬럼만 뽑아서 DeepLearning 표 작성
from new_prep import basic_dt, expand_dt, value_dt,power_dt,clutch_dt,batter_box_dt,basic_war

dfs = [basic_dt, expand_dt, value_dt,power_dt,clutch_dt,batter_box_dt]
f_names = ["basic", 'expaned', 'value', 'power','clutch' ,'batter']

DL_df = pd.DataFrame()

for name,df in zip(f_names, dfs):
    corr_df = pd.read_excel(f'bat_data/{name}_corr.xlsx',index_col=0)
    corr_df = corr_df.loc[corr_df['corr'] >= 0.5]
    col_corr = corr_df.index.to_list()
    col_corr = ['이름','팀'] + col_corr
    if name == 'basic':
        DL_df = df[col_corr]
        continue
    
    DL_df = pd.merge(DL_df,df[col_corr],on=['이름','팀'])
        
DL_df.drop("WAR",axis= 1,inplace=True)
DL_df =  pd.merge(DL_df,basic_war,on=['이름','팀'])
#중복된 컬럼 삭제
DL_df.drop(DL_df.columns[DL_df.columns.str.contains('_y')],axis=1, inplace = True)
DL_df.columns = DL_df.columns.map(lambda x : x[:-2] if "_x" in x else x)    

DL_df.to_excel('DL_df.xlsx')