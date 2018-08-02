# -*- coding:utf8 -*-
#import os
#import csv
import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings('ignore')

#path_train = "/data/dm/train.csv"  # 训练文件
#path_test = "/data/dm/test.csv"  # 测试文件

path_train = "./data/dm/train.csv"  # 训练文件
path_test = "./data/dm/test.csv"  # 测试文件
path_test_out = "./model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

#def read_csv():
#    """
#    文件读取模块，头文件见columns.
#    :return: 
#    """
#    # for filename in os.listdir(path_train):
#    tempdata = pd.read_csv(path_train)
#    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
#                        "CALLSTATE", "Y"]

def f_total_seconds(x):
    return x.total_seconds()

def f_cut(x):
    if x>0:
        return 1
    elif x<0:
        return 2
    else:
        return 0
    
# 急加速、急减速、急刹车    
#def f_a_cut(x):
#    if  x >= 3:
#        return 0
#    elif -3 >= x > -4:
#        return 1
#    elif x<-4:
#        return 2
#    else:
#        return  
    
# 急加速、加速、匀速、减速、急减速、急刹车   
#def f_a_cut(x):
#    if  x >= 3:
#        return 0
#    elif (3 > x > 0):
#        return 1
#    elif x==0:
#        return 2
#    elif (0 > x > -3):
#        return 3
#    elif -3 >= x > -4:
#        return 4
#    else:
#        return 5

# 急加速、加速、匀速、减速、急减速、急刹车   
def f_a_cut(x):
    if  x >= 0.3:
        return 0
    elif (0.3 > x > 0):
        return 1
    elif (0 >= x > -0.2):
        return 2
    elif -0.2 >= x > -0.4:
        return 3
    else:
        return 4

def prod_trian_data(path_train):
    """
    处理数据集
    :return: 
    """
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]

    df=tempdata
    
    df['time_2']=df['TIME'].map(lambda x : datetime.datetime.fromtimestamp(x))
    df.sort_values(['TERMINALNO','TRIP_ID','time_2'],inplace=True)

    df['month_1'] = pd.DatetimeIndex(df['time_2']).month
    df_m=df[['TERMINALNO','month_1']].copy()
    df_m['values']=1
    df_m=df_m.pivot_table(index='TERMINALNO',columns='month_1',values='values', aggfunc=np.sum).fillna(0)
    
    df['dayofweek_1'] = pd.DatetimeIndex(df['time_2']).dayofweek
    df_dw=df[['TERMINALNO','dayofweek_1']].copy()
    df_dw['values']=1
    df_dw=df_dw.pivot_table(index='TERMINALNO',columns='dayofweek_1',values='values', aggfunc=np.sum).fillna(0)    
    
    df['hour_1'] = pd.DatetimeIndex(df['time_2']).hour
    # df['hour_1'] = pd.cut(df['hour_1'],[0,6,9,12,15,18,21,24], labels=["0","1","2","3","4","5","6"]) 
    df_h=df[['TERMINALNO','hour_1']].copy()
    df_h['values']=1
    df_h=df_h.pivot_table(index='TERMINALNO',columns='hour_1',values='values', aggfunc=np.sum).fillna(0)

    df_l2=df[['LONGITUDE','LATITUDE']].copy()
    data=df_l2.values
    # df_l22[['LATITUDE','LONGITUDE']].values[:,0]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans_model=kmeans.fit(data)
    df_l2['kmeans']=list(kmeans_model.predict(data))
    df['kmeans']= df_l2['kmeans']
    # kmeans聚类地理位置
    df_lk=df[['TERMINALNO','kmeans']].copy()
    df_lk['values']=1
    df_lk=df_lk.pivot_table(index='TERMINALNO',columns='kmeans',values='values', aggfunc=np.sum).fillna(0)

    df_l3=df[['TERMINALNO','TRIP_ID','LONGITUDE','LATITUDE']].copy()
    df_l3=df_l3.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).var()
    df_l3=df_l3.fillna(np.mean(df_l3))

    df['DIRECTION_1'] = pd.cut(df['DIRECTION'],[-2,90,180,270,360], labels=["0","1","2","3"])
    df_d=df[['TERMINALNO','DIRECTION_1']].copy()
    df_d['values']=1
    df_d=df_d.pivot_table(index='TERMINALNO',columns='DIRECTION_1',values='values', aggfunc=np.sum)

    df_h2=df[['TERMINALNO','TRIP_ID','HEIGHT']].copy()
    df_h2=df_h2.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).mean()

    df['HEIGHT_diff']=df['HEIGHT'].diff().fillna(0)
    df_h3=df[['TERMINALNO','HEIGHT_diff']].copy()
    df_h3['values']=1
    df_h3['HEIGHT_diff'] = pd.cut(df['HEIGHT_diff'],[-np.Inf,0,np.Inf], labels=["0","1"],right=True)#右边包
    df_h3=df_h3.pivot_table(index='TERMINALNO',columns='HEIGHT_diff',values='values', aggfunc=np.sum)
    
    df_h4=df[['TERMINALNO','TRIP_ID','HEIGHT']].copy()
    df_h4=df_h4.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).var()
    df_h4=df_h4.fillna(np.mean(df_h4))
    
    df_v=df[['TERMINALNO','TRIP_ID','SPEED']].copy()
    df_v=df_v.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).mean().fillna(0)

    df['time_diff']=df.time_2.diff().apply(f_total_seconds).fillna(0).astype('int')
    df['time_diff']=df['time_diff'].map(lambda x:0 if x!=60 else x)
    df['SPEED_diff']=df.SPEED.diff().fillna(0)
    df_s=df[['TERMINALNO','SPEED_diff']].copy()
    df_s['values']=1
    df_s['SPEED_diff_type']= df['SPEED_diff'].apply(f_cut)
    df_s=df_s.pivot_table(index='TERMINALNO',columns='SPEED_diff_type',values='values', aggfunc=np.sum).fillna(0)

    df['SPEED_a']=df['SPEED_diff']/df['time_diff']
    df_a=df[['TERMINALNO','SPEED_a']].copy()
    # df_a = df_a[np.isfinite(df_a)].fillna(0)
    df_a['values']=1
    # df_a['SPEED_a_type']=df_a.SPEED_a.map(lambda x:0 if x>=0 else 1)
    # df_a['SPEED_a_type']=pd.cut(df['SPEED_a'],[-np.Inf,0,np.Inf], labels=["0","1"],right=True)
    df_a['SPEED_a_type']= df['SPEED_a'].apply(f_a_cut)
    df_a=df_a.pivot_table(index='TERMINALNO',columns='SPEED_a_type',values='values', aggfunc=np.sum).fillna(0)


    df_c=df[['TERMINALNO','CALLSTATE']].copy()
    df_c['values']=1
    df_c=df_c.pivot_table(index='TERMINALNO',columns='CALLSTATE',values='values', aggfunc=np.sum).fillna(0)

    data=pd.concat([df_m,df_dw,df_h,df_lk,df_l3,df_d,df_h2,df_h3,df_h4,df_v,df_a,df_c],axis=1)#df_a,,,[海拔], [周几、小时]、[方向]、、[加速度正负、加减速、速度]、[电话]
    data.columns=[str(x) for x in range(1,len(data.columns)+1)] 
    df_y=df[['TERMINALNO','Y']]
    data['Y']=df_y.groupby('TERMINALNO')['Y'].mean()

#    X_train =data.drop(['Y'],axis=1)
#    y_train =data['Y']
#    return X_train,y_train

    data_X=data.drop(['Y'],axis=1)
    data_X=data_X.fillna(0)
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data_X = min_max_scaler.fit_transform(data_X)
    
    data_Y=data['Y']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def prod_val_data(path_test):

    """
    处理数据集
    :return: 
    """
    tempdata = pd.read_csv(path_test)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE"]
    df=tempdata
    
    df['time_2']=df['TIME'].map(lambda x : datetime.datetime.fromtimestamp(x))
    df.sort_values(['TERMINALNO','TRIP_ID','time_2'],inplace=True)

    df['month_1'] = pd.DatetimeIndex(df['time_2']).month
    df_m=df[['TERMINALNO','month_1']].copy()
    df_m['values']=1
    df_m=df_m.pivot_table(index='TERMINALNO',columns='month_1',values='values', aggfunc=np.sum).fillna(0)
    
    df['dayofweek_1'] = pd.DatetimeIndex(df['time_2']).dayofweek
    df_dw=df[['TERMINALNO','dayofweek_1']].copy()
    df_dw['values']=1
    df_dw=df_dw.pivot_table(index='TERMINALNO',columns='dayofweek_1',values='values', aggfunc=np.sum).fillna(0)    
    
    df['hour_1'] = pd.DatetimeIndex(df['time_2']).hour
    # df['hour_1'] = pd.cut(df['hour_1'],[0,6,9,12,15,18,21,24], labels=["0","1","2","3","4","5","6"]) 
    df_h=df[['TERMINALNO','hour_1']].copy()
    df_h['values']=1
    df_h=df_h.pivot_table(index='TERMINALNO',columns='hour_1',values='values', aggfunc=np.sum).fillna(0)

    df_l2=df[['LONGITUDE','LATITUDE']].copy()
    data=df_l2.values
    # df_l22[['LATITUDE','LONGITUDE']].values[:,0]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans_model=kmeans.fit(data)
    df_l2['kmeans']=list(kmeans_model.predict(data))
    df['kmeans']= df_l2['kmeans']
    # kmeans聚类地理位置
    df_lk=df[['TERMINALNO','kmeans']].copy()
    df_lk['values']=1
    df_lk=df_lk.pivot_table(index='TERMINALNO',columns='kmeans',values='values', aggfunc=np.sum).fillna(0)

    df_l3=df[['TERMINALNO','TRIP_ID','LONGITUDE','LATITUDE']].copy()
    df_l3=df_l3.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).var()
    df_l3=df_l3.fillna(np.mean(df_l3))        
    
    df['DIRECTION_1'] = pd.cut(df['DIRECTION'],[-2,90,180,270,360], labels=["0","1","2","3"])
    df_d=df[['TERMINALNO','DIRECTION_1']].copy()
    df_d['values']=1
    df_d=df_d.pivot_table(index='TERMINALNO',columns='DIRECTION_1',values='values', aggfunc=np.sum)

    df_h2=df[['TERMINALNO','TRIP_ID','HEIGHT']].copy()
    df_h2=df_h2.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).mean()

    df['HEIGHT_diff']=df['HEIGHT'].diff().fillna(0)
    df_h3=df[['TERMINALNO','HEIGHT_diff']].copy()
    df_h3['values']=1
    df_h3['HEIGHT_diff'] = pd.cut(df['HEIGHT_diff'],[-np.Inf,0,np.Inf], labels=["0","1"],right=True)#右边包
    df_h3=df_h3.pivot_table(index='TERMINALNO',columns='HEIGHT_diff',values='values', aggfunc=np.sum)

    df_h4=df[['TERMINALNO','TRIP_ID','HEIGHT']].copy()
    df_h4=df_h4.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).var()
    df_h4=df_h4.fillna(np.mean(df_h4))
    
    df_v=df[['TERMINALNO','TRIP_ID','SPEED']].copy()
    df_v=df_v.groupby(['TERMINALNO','TRIP_ID']).mean().groupby(['TERMINALNO']).mean()

    df['time_diff']=df.time_2.diff().apply(f_total_seconds).fillna(0).astype('int')
    df['time_diff']=df['time_diff'].map(lambda x:0 if x!=60 else x)
    df['SPEED_diff']=df.SPEED.diff().fillna(0)
    df_s=df[['TERMINALNO','SPEED_diff']].copy()
    df_s['values']=1
    df_s['SPEED_diff_type']= df['SPEED_diff'].apply(f_cut)
    df_s=df_s.pivot_table(index='TERMINALNO',columns='SPEED_diff_type',values='values', aggfunc=np.sum)

    df['SPEED_a']=df['SPEED_diff']/df['time_diff']
    df_a=df[['TERMINALNO','SPEED_a']].copy()
    # df_a = df_a[np.isfinite(df_a)].fillna(0)
    df_a['values']=1
    # df_a['SPEED_a_type']=df_a.SPEED_a.map(lambda x:0 if x>=0 else 1)
    # df_a['SPEED_a_type']=pd.cut(df['SPEED_a'],[-np.Inf,0,np.Inf], labels=["0","1"],right=True)
    df_a['SPEED_a_type']= df['SPEED_a'].apply(f_a_cut)
    df_a=df_a.pivot_table(index='TERMINALNO',columns='SPEED_a_type',values='values', aggfunc=np.sum)
    
    print(df['SPEED_a'].apply(f_a_cut).value_counts())
    df_c=df[['TERMINALNO','CALLSTATE']].copy()
    df_c['values']=1
    df_c=df_c.pivot_table(index='TERMINALNO',columns='CALLSTATE',values='values', aggfunc=np.sum).fillna(0)

    X_val=pd.concat([df_m,df_dw,df_h,df_lk,df_l3,df_d,df_h2,df_h3,df_h4,df_v,df_a,df_c],axis=1)#[海拔], [周几、小时]、[方向]、、[加速度正负、加减速、速度]、[电话]
    X_val.columns=[str(x) for x in range(1,len(X_val.columns)+1)] 
    X_val_df=X_val.fillna(0)
    
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    
    X_val= min_max_scaler.fit_transform(X_val_df)
    
    return X_val,X_val_df


#def gini(actual, pred):
#    assert (len(actual) == len(pred))
#    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
#    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
#    totalLosses = all[:, 0].sum()
#    giniSum = all[:, 0].cumsum().sum() / totalLosses
#
#    giniSum -= (len(actual) + 1) / 2.
#    return giniSum / len(actual)
#
#def gini_normalized(actual, pred):
#    return gini(actual, pred) / gini(actual, actual)

def model():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
#    import xgboost as xgb
#    dtrain=xgb.DMatrix(X_train, label=y_train)
#    dtest=xgb.DMatrix(X_test, label=y_test)
#    dval = xgb.DMatrix(X_val)
#    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
#        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
#        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'reg:linear'}
#    num_round = 283
#    param['nthread'] = 4
#    param['eval_metric'] = "auc"
#    param.update({'eval_metric': 'logloss'})
#    plst = param.items()
#    evallist = [(dtest, 'eval'), (dtrain, 'train')]
#    xgbr=xgb.train(plst, dtrain, num_round, evallist)
    


    


#    from sklearn.model_selection import GridSearchCV
##    import xgboost as xgb
#    from xgboost.sklearn import XGBRegressor
#    
#    cv_params = { 'max_depth':list(range(10,2,-1)),'min_child_weight':list(range(6,1,-1)}
#    other_params = {'learning_rate': 0.1, 'seed': 0, 'n_estimators': 500,'subsample': 0.8,
#                    'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#
#    model = XGBRegressor(**other_params)
#    xgbr = GridSearchCV(estimator=model, param_grid=cv_params, cv=5, verbose=1, n_jobs=4)
#    xgbr.fit(X_train, y_train)
#    print('每轮迭代运行结果:{0}'.format(xgbr.grid_scores_))
#    print('参数的最佳取值：{0}'.format(xgbr.best_params_))
#    print('最佳模型得分:{0}'.format(xgbr.best_score_))
#    from sklearn.ensemble import RandomForestRegressor
#    xgbr = RandomForestRegressor()
#    xgbr.fit(X_train, y_train)

    from xgboost import XGBRegressor
    xgbr = XGBRegressor(max_depth=4)
    print(xgbr)
    xgbr.fit(X_train, y_train)
    
    import lightgbm as lgb
    lgbr = lgb.LGBMRegressor(max_depth=6) 
    print(lgbr)
    lgbr.fit(X_train, y_train)


#    xgbr = XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
#                         learning_rate=0.1, max_delta_step=0, max_depth=5,
#                         min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#                         objective='reg:linear', reg_alpha=0, reg_lambda=1,
#                         scale_pos_weight=1, seed=0, silent=True, subsample=1)
  
#    from xgboost import plot_importance  
#    from matplotlib import pyplot  
#    plot_importance(xgbr,importance_type = 'cover')  
#    pyplot.show() 
    
#    from sklearn import preprocessing 
#    Pred = preprocessing.scale(xgbr.predict(X_val))
#    Pred = xgbr.predict(X_val)
#    Pred=(xgbr.predict(X_val)-xgbr.predict(X_val).min())/((xgbr.predict(X_val).max()-xgbr.predict(X_val).min()))
#    prep_1=np.log(xgbr.predict(X_val))
    
    Id_pred=pd.DataFrame()
    Id_pred['Id']=X_val_df.index
    Id_pred['pred_1']=xgbr.predict(X_val)
    Id_pred['pred_2']=lgbr.predict(X_val)
    
    
#    Id_pred['Pred']=prep_1
    Id_pred['Pred']=0.6*Id_pred['pred_1'].rank()+0.4*Id_pred['pred_2'].rank()
    
    del Id_pred['pred_1'],Id_pred['pred_2']
    Id_pred.to_csv(path_test_out+"test.csv",index=None)
#    print(Id_pred['Pred'])#.value_counts().sort_values()
    
    
    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
    print ('The score is:', xgbr.score(X_test, y_test))
    print('The r2_score is:', r2_score(y_test,xgbr.predict(X_test)))    
    print('The mean_squared_error is:', mean_squared_error(y_test,xgbr.predict(X_test)))
    print('The mean_absolute_error is:', mean_absolute_error(y_test,xgbr.predict(X_test)))
   
#    print(xgbr.booster().get_score())
#    from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#    print ('The score is:', xgbr.score(X_test, y_test))
#    print('The r2_score is:', r2_score(y_test,xgbr.predict(X_test)))    
#    print('The mean_squared_error is:', mean_squared_error(y_test,xgbr.predict(X_test)))
#    print('The mean_absolute_error is:', mean_absolute_error(y_test,xgbr.predict(X_test)))

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
#    X_train, y_train = prod_trian_data(path_train)
    X_train, X_test, y_train, y_test = prod_trian_data(path_train)
    X_val,X_val_df=prod_val_data(path_test)
    model()