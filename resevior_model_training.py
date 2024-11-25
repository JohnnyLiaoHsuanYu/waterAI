import pickle
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBClassifier
from joblib import dump, load


resevior_rd_path = 'D:/2310011_Liao/水資源/訓練資料/resevior/'
train_name = 'predict_resevior_percentage_withname.csv'
df = pd.read_csv(resevior_rd_path + train_name)

need_resevior_name = ['仁義潭水庫', '南化水庫', '寶山水庫', '寶山第二水庫', '德基水庫', '新山水庫', '日月潭水庫', '明德水庫', \
                 '曾文水庫', '永和山水庫', '湖山水庫', '烏山頭水庫', '牡丹水庫', '石門水庫', '翡翠水庫', '蘭潭水庫', \
                 '阿公店水庫', '鯉魚潭水庫']

need_resevior_name_2 = ['澄清湖水庫', '石岡壩', '集集攔河堰', '鳳山水庫']

# with open('D:/2310011_Liao/水資源/orange/model/resevior_random_forest.pkcls', 'rb') as f:
#     modell = pickle.load(f)


# MODEL_ORG = RandomForestRegressor(n_estimators = 10, criterion = 'squared_error', max_depth = 10, min_samples_split = 6, min_samples_leaf = 5, \
#                                   min_weight_fraction_leaf = 0.0, max_features = 1.0, max_leaf_nodes = None, bootstrap = True, oob_score = False, \
#                                   n_jobs = 1, random_state = None, verbose = 0,
#                                   )

# MODEL_ORG = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=11, min_samples_split=1000, 
#                                min_samples_leaf=500, min_weight_fraction_leaf=0.0, max_features=None, random_state=42, 
#                                max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)

MODEL_ORG = XGBRegressor(max_depth = 10, learning_rate = 0.18, n_estimators = 10, 
                     verbosity = None, objective = 'reg:squarederror', booster = None, 
                     tree_method = None, n_jobs = None, gamma = None, min_child_weight = None, 
                     max_delta_step = None, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, 
                     colsample_bynode = 1, reg_alpha = None, reg_lambda = 1, scale_pos_weight = None, 
                     base_score = None, random_state = 0, num_parallel_tree = None, 
                     monotone_constraints = None, interaction_constraints = None, importance_type = 'gain', 
                     gpu_id = None, validate_parameters = None)

# MODEL_ORG = MLPRegressor(hidden_layer_sizes = (8, 16, 32, 16, 8), activation = 'relu', solver = 'adam', alpha = 0.001, 
#                      batch_size = 'auto', learning_rate = 'adaptive', learning_rate_init = 0.004, power_t = 0.5, 
#                      max_iter = 100, shuffle = True, random_state = 1, tol = 0.0001, verbose = False, warm_start = False, 
#                      momentum = 0.9, nesterovs_momentum = True, early_stopping = True, validation_fraction = 0.1, 
#                      beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)


RRR = []
for i in need_resevior_name:
    filter_name = (df['reseviorname']==i)
    df_specify = df.loc[filter_name]

    X = df_specify.iloc[:, 1:5]
    y = df_specify['percentage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    MODEL_ = MODEL_ORG
    
    MODEL_.fit(X_train, y_train)
    MODEL_pred = MODEL_.predict(X_test)

    mse = mean_squared_error(y_test, MODEL_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, MODEL_pred)
    r2 = r2_score(y_test, MODEL_pred)

    RRR.append({'name': i, 'R-score': r2*100})

    print(' ')
    print('===================================')
    print(i)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2*100} %')
    print(' ')


    if MODEL_.__class__.__name__ == 'MLPRegressor':
        result = permutation_importance(MODEL_, X, y, n_repeats=10, random_state=42)
        feature_importances = {}
        for fff in result.importances_mean.argsort()[::-1]:
            feature_importances[X.columns[fff]] = np.round(result.importances_mean[fff], 5)
            # print(f"Feature {X.columns[i]}: {result.importances_mean[i]:.4f}")
    else:
        importances = MODEL_.feature_importances_
        feature_names = X.columns
        feature_importances = {}
        for ff in range(len(feature_names)):
            feature_importances[feature_names[ff]] = np.round(importances[ff]*100, 4)

    print(feature_importances)
    print(' ')
    print('===================================')

    # # Cross-validation scores
    # # Set up cross-validation
    # print('Cross-validation')
    # kf = KFold(n_splits=15, shuffle=True, random_state=42)
    # acc_score = []
    # count = 0
    # for train_index, test_index in kf.split(X, y):
    #     X_train_kf = X.iloc[train_index, :]
    #     X_test_kf = X.iloc[test_index, :]
    #     y_train_kf = y.iloc[train_index]
    #     y_test_kf = y.iloc[test_index]

    #     RFC_2 = RandomForestRegressor(n_estimators = 20,
    #                                 criterion = 'squared_error',
    #                                 max_depth = None,
    #                                 min_samples_split = 3,
    #                                 min_samples_leaf = 1,
    #                                 min_weight_fraction_leaf = 0.0,
    #                                 max_features = 1.0,
    #                                 max_leaf_nodes = None,
    #                                 bootstrap = True, # 隨機抽樣
    #                                 oob_score = False,
    #                                 n_jobs = 1, 
    #                                 random_state = None,
    #                                 verbose = 0,
    #                                 )

    #     RFC_2.fit(X_train_kf, y_train_kf)

    #     RFC_re_pred_kf = RFC_2.predict(X_test_kf)

    #     mse_kf = mean_squared_error(y_test_kf, RFC_re_pred_kf)
    #     rmse_kf = np.sqrt(mse_kf)
    #     mae_kf = mean_absolute_error(y_test_kf, RFC_re_pred_kf)
    #     r2_kf = r2_score(y_test_kf, RFC_re_pred_kf)
    #     count += 1
    #     acc_score.append(r2_kf)
    #     # print(' ')
    #     # print(i)
    #     # print(f'{count} R² Score: {r2_kf*100} %')
    
    # acc_score_mean = np.nanmean(np.array(acc_score))
    # print(i,': Cross-validation mean R score', acc_score_mean*100, '%')

    today = datetime.date.today()
    formatted_today = today.strftime('%Y%m%d')
    os.makedirs('D:/2310011_Liao/水資源/done_model/resevior/', exist_ok=True)
    dump(MODEL_, 'D:/2310011_Liao/水資源/done_model/resevior/resevior_%s_%s.joblib'%(MODEL_.__class__.__name__, i)) #_withoutresevior




MODEL_ORG_2 = XGBRegressor(max_depth = 10, learning_rate = 0.4, n_estimators = 10, 
                     verbosity = None, objective = 'reg:squarederror', booster = None, 
                     tree_method = None, n_jobs = None, gamma = None, min_child_weight = None, 
                     max_delta_step = None, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, 
                     colsample_bynode = 1, reg_alpha = None, reg_lambda = 1, scale_pos_weight = None, 
                     base_score = None, random_state = 0, num_parallel_tree = None, 
                     monotone_constraints = None, interaction_constraints = None, importance_type = 'gain', 
                     gpu_id = None, validate_parameters = None)

RRR2 = []
for i in need_resevior_name_2:
    filter_name = (df['reseviorname']==i)
    df_specify = df.loc[filter_name]

    X = df_specify.iloc[:, 1:5]
    y = df_specify['percentage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    MODEL_2 = MODEL_ORG_2
    
    MODEL_2.fit(X_train, y_train)
    MODEL_2_pred = MODEL_2.predict(X_test)

    mse = mean_squared_error(y_test, MODEL_2_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, MODEL_2_pred)
    r2 = r2_score(y_test, MODEL_2_pred)

    RRR2.append({'name': i, 'R-score': r2*100})

    print(' ')
    print('===================================')
    print(i)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2*100} %')
    print(' ')


    if MODEL_2.__class__.__name__ == 'MLPRegressor':
        result = permutation_importance(MODEL_, X, y, n_repeats=10, random_state=42)
        feature_importances = {}
        for fff in result.importances_mean.argsort()[::-1]:
            feature_importances[X.columns[fff]] = np.round(result.importances_mean[fff], 5)
            # print(f"Feature {X.columns[i]}: {result.importances_mean[i]:.4f}")
    else:
        importances = MODEL_2.feature_importances_
        feature_names = X.columns
        feature_importances = {}
        for ff in range(len(feature_names)):
            feature_importances[feature_names[ff]] = np.round(importances[ff]*100, 4)

    print(feature_importances)
    print(' ')
    print('===================================')

    today = datetime.date.today()
    formatted_today = today.strftime('%Y%m%d')
    os.makedirs('D:/2310011_Liao/水資源/done_model/resevior/', exist_ok=True)
    dump(MODEL_2, 'D:/2310011_Liao/水資源/done_model/resevior/resevior_%s_%s.joblib'%(MODEL_2.__class__.__name__, i)) #_withoutresevior


# R_small = np.where(np.array(RRR)<90)[0]
# for rrr in R_small:
#     print(need_resevior_name[rrr], 'R²', RRR[rrr], '%')

