'''
Algorithm: XGBoost
Purpose:  0-1 m SOC
Author: T.Shen
Date: 2023-10-10
Paper Title:
"Frozen carbon is gradually thawing: Assessing interannual dynamics of thawed soil organic carbon stocks in the Tibetan Plateau permafrost area from 1901−2020"
Journal: Agricultural and Forest Meteorology
'''

# Import library
import numpy as np
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

Train_dataset="F:\Paper3-Basic data-C+N\SOC_Characteristic/4.Train_Baseline_Data\Train_PreviousStudy/Train1_0-1m_Final.csv"

Predict_Time_Periods=['F:\Paper3-Basic data-C+N\SOC_Characteristic/3.Predicted_Baseline_SOC/Predicted_Baseline_SOC_0-1m.csv']

Output_Time_Periods=['F:/Paper3-Basic data-C+N/Text/SOC_XGB_0-1m.txt']

Index_dataset='F:/2.中间数据暂时储存/Combination/lon-lat/indexes_txt.csv'

data_PF_train = np.genfromtxt(Train_dataset, skip_header=1, delimiter=",")
x_data_order = data_PF_train[:,:-1]
y_data_order = data_PF_train[:,-1]
x_data, y_data = sklearn.utils.shuffle(x_data_order, y_data_order, random_state=10)

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
    n_estimators=136,
    learning_rate=0.09,
    gamma=92,
    max_depth=7,
    min_child_weight=7,
    max_delta_step=2,
    subsample=1,
    colsample_bytree=0.9,
    colsample_bylevel=1,
#    colsample_bynote=1,
    #scale_pos_weight=0.2,
    reg_alpha=21,
    reg_lambda=0.5,
    objective='reg:squarederror',
    base_score=0.5,
    eval_metric=['rmse','mae','auc'],
    seed=0,
    use_label_encoder=False)
    #tree_method='gpu_hist',
    #gpu_id=0)

hint_calibration = input('Parameter calibration method, Method_1-Cross-validation (1) / Method_2-Divide the training/testing set (2) / Method_3-No calibration parameters (any key):')
if hint_calibration == '1':
    cv_params = {
        #'n_estimators':[100,110,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,145,150,155,160,165,170,175,180,190,200,300,400,],
        #'learning_rate': [0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.3,0.4,0.5,0.6],
        #'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        #'min_child_weight': [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15],
        #'max_delta_step': [0, 0.2,0.4, 0.6, 0.8, 1, 2,3,4,5,6,7,8,9,10],
        #'subsample': [0.6, 0.7, 0.8,0.85, 0.9,0.95, 1],
        #'colsample_bytree': [0.5, 0.6, 0.7,0.75,  0.8, 0.85, 0.9,1],
        #'reg_alpha': [0,1,3,5,7,10,15,16,17,18,19,20,21,22,23,24,25,30,40,50,60,65,70,80,90,100],
        #'reg_lambda': [0,0.1,0.2,0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3,4,5,10,20,30,40,50,60,70,80,90,100],
        #'gamma': [0, 0.1, 0.2, 0.3,0.4, 0.5,1, 10,20,30,50,60,70,80,90,91,92,93,94,95,96,97,98,99,100],
        #'scale_pos_weight': [0,0.1,0.2,0.3, 0.4,0.5, 0.6, 0.8, 1,1.2,1.3,1.4,2,3,4],
        #'colsample_bylevel':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    }
    gsearch = GridSearchCV(XGB_R, param_grid=cv_params, scoring='r2', cv=10)
    gsearch.fit(x_train, y_train)

    '''
    verification=cross_val_predict(XGB_R, x_train, y_train, cv=10)
    np.savetxt("F:\Paper3-Basic data-C+N\SOC_Characteristic/5.Result\Verification_SOC/XGB_SOC_V_0-1.csv",
               np.hstack((y_train.reshape(-1,1), verification.reshape(-1,1))), delimiter=",")'''

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
    print(score1)
    print(score2)
else:
    print('Unable to continue')


hint_prediction = input('Prediction or not, please enter y/n:')

if hint_prediction == 'y':
    for i in range(1):

        Predict_dataset = Predict_Time_Periods[int(i)]
        Output_dataset = Output_Time_Periods[int(i)]

        data_PF_predict=np.genfromtxt(Predict_dataset, delimiter=",")

        x_predict = data_PF_predict[:,:]
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