'''
Algorithm: XGBoost
Purpose:  E-Factor
Author: T.Shen
Date: 2023-10-10
Paper Title:
"Frozen carbon is gradually thawing: Assessing interannual dynamics of thawed soil organic carbon stocks in the Tibetan Plateau permafrost area from 1901−2020"
Journal: Agricultural and Forest Meteorology
'''

# Import library
import numpy as np
import os
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import time
import sklearn

time_start=time.time()

def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


Train_dataset="F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/5.Train-data\组合数据/ALT_E_Train_Final2.csv"

Predict_Path = 'F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/6.Predicted data\Predicted_Baseline_PF'
path_list= os.listdir(Predict_Path)

Output_Path = 'F:/Paper3-Basic data-C+N/Text'

Index_dataset='F:/2.中间数据暂时储存/Combination/lon-lat/indexes_txt.csv'

data_PF_train = np.genfromtxt(Train_dataset, skip_header=1, delimiter=",")
z_data2=data_PF_train[:,:-1]
x_data_order = data_PF_train[:,:-2]
y_data_order = data_PF_train[:,-1]
x_data, y_data,z_data = sklearn.utils.shuffle(x_data_order, y_data_order,z_data2, random_state=10)


# Divide the training and test sets
hint_split=input('Whether to divide the training and test sets, please enter y/n:')

if hint_split == 'y':
    x_train,x_test,y_train,y_test = train_test_split(x_data, y_data, test_size = 0.8)
else:
    x_train = x_data
    y_train = y_data

# Model hyperparameters
XGB_R=XGBRegressor(
    booster='gbtree',
    nthread=10,
    n_estimators=180,
    learning_rate=0.07,
    gamma=0,
    max_depth=10,
    min_child_weight=3,
    max_delta_step=0.2,
    subsample=1,
    colsample_bytree=0.9,
    colsample_bylevel=0.5,
#    colsample_bynote=1,
    scale_pos_weight=0,
    reg_alpha=0.13,
    reg_lambda=0.71,
    objective='reg:squarederror',
    base_score=0.5,
    eval_metric=['rmse','mae','auc'],
    use_label_encoder=False)

# Parameter calibration method
hint_calibration = input('Parameter calibration method, Method_1-Cross-validation (1) / Method_2-Divide the training/testing set (2) / Method_3-No calibration parameters (any key):')
if hint_calibration == '1':
    cv_params = {
        #'n_estimators':[100,110,120,130,140,150,160,170,175,176,177,178,179,180,181,182,183,184,185,190,195,196,197,198,199,200,201,202,203,204,205,210],
        #'learning_rate': [0.01,0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.11,0.12,0.13],
        #'max_depth': [5,6,7,8,9,10,11,12,13],
        #'min_child_weight': [0, 1, 2, 3, 4, 5,6,],
        #'max_delta_step': [0, 0.1,0.2,0.4, 0.5,0.6, 0.7,0.8,],
        #'subsample': [0.4,0.5,0.6, 0.7, 0.8,0.9,1],
        #'colsample_bytree': [0.5, 0.6, 0.7, 0.8,  0.9,1],
        #'colsample_bylevel':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        #'reg_alpha': [0,0.05, 0.08, 0.09, 0.1, 0.11, 0.12,0.13,0.14, 0.15,  0.2,], #[0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1,10,20,30],
        #'reg_lambda': [0.55,0.6,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.8,0.9,], #[0.6,0.7,0.8,0.9,1,10,20,30],
        #'gamma': [0,0.01,0.03, 0.1, 0.2, 0.3,],
        #'scale_pos_weight': [0, 0.1,0.2, 0.3,0.4, 0.6, 0.8, 1,1.2,1.3,1.4,2,3,4,5,6,7,10,20,30]
    }
    gsearch = GridSearchCV(XGB_R, param_grid=cv_params, scoring='r2', cv=10)
    gsearch.fit(x_train, y_train)

    '''
    verification=cross_val_predict(XGB_R, x_train, y_train, cv=10)
    np.savetxt("F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/7.Result_New\Ver_ALT/ALT_XGB_Verif.csv",
               np.hstack((z_data, y_train.reshape(-1,1), verification.reshape(-1,1))), delimiter=",")'''

    print('Optimal parameter: {0}'.format(gsearch.best_params_))
    #print('Score: {0}'.format(gsearch.best_score_))

elif hint_calibration == '2':
    if hint_split == 'y':
        XGB_R.fit(x_train, y_train)
        y_pred = XGB_R.predict(x_test)
        score = metrics.mean_squared_error(y_test, y_pred)
        print(score)
    else:
        print('No division between training and test sets')
else:
    print('No calibration parameters')

hint_all_train = input('Whether to use all data to train the final model, please enter y/n:')
if hint_all_train == 'y':
    XGB_R.fit(x_data, y_data)
    y_pred1 = XGB_R.predict(x_data)
    score1 = metrics.mean_squared_error(y_data, y_pred1)
    score2 = metrics.r2_score(y_data, y_pred1)
    print(score1) #'均方根误差：'+
    print(score2) #'r2：' +
else:
    print('Unable to continue')


hint_prediction = input('Prediction or not, please enter y/n:')

if hint_prediction == 'y':
    for file in path_list:
        Predict_dataset = os.path.join(Predict_Path, file)

        Outfile = 'ALT_XGB_E.txt'
        Output_dataset = os.path.join(Output_Path, Outfile)

        data_PF_predict=np.genfromtxt(Predict_dataset, delimiter=",")

        x_predict = data_PF_predict[:, :]
        y_predict = XGB_R.predict(x_predict)

        indexes = np.genfromtxt(Index_dataset, skip_header=1, delimiter=",")
        index_list = indexes[:, -1]

        j = 0
        ASCII_list = []
        for i in index_list:
            if i == 0:
                ASCII_list.append(-9999)
            else:
                ASCII_list.append(y_predict[j])
                j += 1

        ASCII_list_split = func(ASCII_list, 2821)

        with open(Output_dataset, 'w') as f:
            f.write('ncols         2821' + '\n')
            f.write('nrows         1729' + '\n')
            f.write('xllcorner     -664833.32935843' + '\n')
            f.write('yllcorner     2865493.7575816' + '\n')
            f.write('cellsize      1000' + '\n')
            f.write('NODATA_value  -9999' + '\n')
            for line_list in ASCII_list_split:
                for j in line_list:
                    f.write(str(j) + ' ')
                f.write('\n')
else:
    print('Unfortunately, no predictions were made')

time_end=time.time()
print('time cost',time_end-time_start,'s')