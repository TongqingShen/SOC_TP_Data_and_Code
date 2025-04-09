'''
Algorithm: LightGBM
Purpose:  Permafrost distribution
Author: T.Shen
Date: 2023-10-10
Paper Title:
"Frozen carbon is gradually thawing: Assessing interannual dynamics of thawed soil organic carbon stocks in the Tibetan Plateau permafrost area from 1901−2020"
Journal: Agricultural and Forest Meteorology
'''

# Import library
import numpy as np
from lightgbm import LGBMClassifier
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


Train_dataset="F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/5.Train-data\组合数据/PF-train.csv" # Train Data

Predict_Time_Periods=['F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/6.Predicted data\Predicted_Baseline_PF/Predicted_Baseline.csv'] # Prediction data

Output_Time_Periods=['F:/Paper3-Basic data-C+N/Text/PF_Baseline_LGB.txt'] # Output

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
LGB = LGBMClassifier(
    boosting_type = 'gbdt',
    objective = 'binary',
    n_estimators = 300,
    learning_rate = 0.08,
    min_split_gain = 0,
    min_child_samples = 22,
    min_child_weight = 0.001,
    max_depth = 4,
    num_leaves = 16,
    subsample = 0.1,
    subsample_freq = 0,
    colsample_bytree = 0.2,
    reg_alpha = 0.6,
    reg_lambda = 0.3,
    class_weight= None,
    random_state = None,
    n_jobs = 10,
    importance_type = 'split')

# Parameter calibration method
hint_calibration = input('Parameter calibration method, Method_1-Cross-validation (1) / Method_2-Divide the training/testing set (2) / Method_3-No calibration parameters (any key):')
if hint_calibration == '1':
    cv_params = {
        #'n_estimators':[295,296,297,298,299,300,301,302,303,304,305],
        #'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.07, 0.08,0.09,0.1,0.15,0.2,0.3,0.4,0.5],
        #'min_child_samples': [15,16,17,18,19,20,21,22,23,24,25,40,50,60],
        #'min_child_weight': [0.00001,0.0001,0.001, 0.002, 0.003, 0.005, 0.01, 0.015,0.02,0.1,],
        #'max_depth': [1, 2, 3, 4,5,6,7,8,9,10],
        #'num_leaves': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25],
        #'subsample': [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9,1],
        #'colsample_bytree': [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9,1],
        #'reg_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 10, 20, 50],
        #'reg_lambda': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 5, 10, 20, 50],
    }
    gsearch = GridSearchCV(LGB, param_grid=cv_params, scoring='accuracy', cv=10)
    gsearch.fit(x_train, y_train)

    '''
    verification=cross_val_predict(LGB, x_train, y_train, cv=10)
    np.savetxt("F:\Paper3-Basic data-C+N\ALT_PF_Characteristic/7.Result_New\Ver_PF/PF_LGB.csv",
               np.hstack((y_train.reshape(-1,1), verification.reshape(-1,1))), delimiter=",")'''

    print('Optimal parameter: {0}'.format(gsearch.best_params_))
    #print('Score: {0}'.format(gsearch.best_score_))

elif hint_calibration == '2':
    if hint_split == 'y':
        LGB.fit(x_train, y_train)
        y_pred = LGB.predict(x_test)
        score = metrics.mean_squared_error(y_test, y_pred)
        print(score)
    else:
        print('No division between training and test sets')

else:
    print('No calibration parameters')


hint_all_train = input('Whether to use all data to train the final model, please enter y/n:')
if hint_all_train == 'y':
    LGB.fit(x_data, y_data)
    print(LGB.score(x_data, y_data))
else:
    print('Unable to continue')


hint_prediction = input('Prediction or not, please enter y/n:')

if hint_prediction == 'y':
    for i in range(1):

        Predict_dataset = Predict_Time_Periods[int(i)]
        Output_dataset = Output_Time_Periods[int(i)]

        data_PF_predict=np.genfromtxt(Predict_dataset, delimiter=",")

        x_predict =  data_PF_predict[:, :]
        y_predict = LGB.predict(x_predict)

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