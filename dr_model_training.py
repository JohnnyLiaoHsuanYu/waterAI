import os
import pickle
import pandas as pd
import numpy as np
import sys
import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.cluster import KMeans
from sklearn import svm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from subprocess import call
from sklearn.inspection import permutation_importance
from joblib import dump, load
from IPython.display import Image

from lib.log_module import CustomLog



log_path = os.path.join("D:", "2310011_Liao", "水資源" , "done_model", "drought", "logg", "dr_model.log")
logger = CustomLog(log_path, "Fuck")


start_time = time.time()
# input data
model_path = '/home/ubuntu/model_data/'
rd_path = 'D:/2310011_Liao/水資源/訓練資料/drought/'
filename = 'training_data_20241119.csv' #6_withoutresevior
df = pd.read_csv(rd_path + filename)
df = df.dropna()


X = df.drop('level',axis=1)
y = df['level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


######## Regression model ########

# MODEL_ORG = MLPRegressor(hidden_layer_sizes = (16, 32, 64, 32, 16), activation = 'relu', solver = 'adam', alpha = 0.001, 
#                      batch_size = 'auto', learning_rate = 'adaptive', learning_rate_init = 0.06, power_t = 0.5, 
#                      max_iter = 100, shuffle = True, random_state = 1, tol = 0.0001, verbose = False, warm_start = False, 
#                      momentum = 0.9, nesterovs_momentum = True, early_stopping = True, validation_fraction = 0.1, 
#                      beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)



# MODEL_ORG = DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=11, min_samples_split=1000, 
#                                min_samples_leaf=500, min_weight_fraction_leaf=0.0, max_features=None, random_state=42, 
#                                max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0)
                               

# MODEL_ORG = RandomForestRegressor(n_estimators=10, criterion='squared_error', max_depth=10, min_samples_split=1000, 
#                               min_samples_leaf=500, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, 
#                               min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, 
#                               verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)


MODEL_ORG = XGBRegressor(max_depth = 8, learning_rate = 0.3, n_estimators = 10, 
                     verbosity = None, objective = 'reg:squarederror', booster = None, 
                     tree_method = None, n_jobs = None, gamma = None, min_child_weight = None, 
                     max_delta_step = None, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, 
                     colsample_bynode = 1, reg_alpha = None, reg_lambda = 1, scale_pos_weight = None, 
                     base_score = None, random_state = 0, num_parallel_tree = None, 
                     monotone_constraints = None, interaction_constraints = None, importance_type = 'gain', 
                     gpu_id = None, validate_parameters = None)



MODEL_ = MODEL_ORG

# fitting
MODEL_.fit(X_train, y_train)
MODEL_pred = MODEL_.predict(X_test)

mse = mean_squared_error(y_test, MODEL_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, MODEL_pred)
r2 = r2_score(y_test, MODEL_pred)


y_train_pred = MODEL_.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)


print(f'{MODEL_.__class__.__name__}')
print('=========================')
print('TRAINING DATA:')
print('MSE:', train_mse)
print('R2:', np.round(train_r2*100, 3))
print('=========================')
print('TESTING DATA:')
print('MSE:', mse)
print('R2:', np.round(r2*100, 3))


if MODEL_.__class__.__name__ == 'MLPRegressor':
    result = permutation_importance(MODEL_, X, y, n_repeats=10, random_state=42)
    feature_importances = {}
    for i in result.importances_mean.argsort()[::-1]:
        feature_importances[X.columns[i]] = np.round(result.importances_mean[i], 5)
        print(f"Feature {X.columns[i]}: {result.importances_mean[i]:.4f}")
else:
    importances = MODEL_.feature_importances_
    feature_names = X.columns
    feature_importances = {}
    for ff in range(len(feature_names)):
        feature_importances[feature_names[ff]] = np.round(importances[ff]*100, 4)



# if MODEL_.__class__.__name__ == 'DecisionTreeRegressor':
#     # Export as dot file
#     export_graphviz(MODEL_, 
#                     out_file='D:/2310011_Liao/水資源/done_model/drought/tree_decision_regressor.dot', 
#                     feature_names = X.columns,
#                     #class_names=,
#                     rounded = True, 
#                     proportion = False, 
#                     precision = 2, 
#                     filled = True)
#     # Convert to png using system command (requires Graphviz)
#     call(['D:', '2310011_Liao', '水資源', 'done_model', 'drought', 'dot', '-Tpng', 'tree_decision_regressor.dot', '-o', 'tree_decision_regressor.png', '-Gdpi=600', '-Epenwidth=3'])
#             #,'-Gsize=30,4!', '-Gratio=fill', '-Nwidth=4', '-Nheight=3', '-Nfontsize=50', '-Epenwidth=9'])
#     # Display in jupyter notebook
#     Image(filename = 'D:/2310011_Liao/水資源/done_model/drought/tree_decision_regressor.png')
    
    
#     param_grid = {'max_leaf_nodes': [5, 10, 20, 30, 50, 100, None]}
#     grid_search = GridSearchCV(MODEL_, param_grid, cv=5, scoring='r2')
#     grid_search.fit(X_train, y_train)
    
#     print("Best max_leaf_nodes:", grid_search.best_params_['max_leaf_nodes'])
#     print("Best R2:", (grid_search.best_score_)*100, '%')
  

logger.make_log(f'\n\
                ===================================\n\
                {MODEL_.__class__.__name__}\n\
                Mean Squared Error (MSE): {mse}\n\
                Root Mean Squared Error (RMSE): {rmse}\n\
                Mean Absolute Error (MAE): {mae}\n\
                R² Score: {r2*100} %\n\
                {feature_importances}'
                )
logger.log.rm_handler()



print('Cross-validation scores')
# Cross-validation scores
# Set up cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_score = []
mmmnnn = 0
for train_index, test_index in skf.split(X, y):
    X_train_skf = X.iloc[train_index, :]
    X_test_skf = X.iloc[test_index, :]
    y_train_skf = y.iloc[train_index]
    y_test_skf = y.iloc[test_index]

    MODEL_2 = MODEL_ORG

    MODEL_2.fit(X_train_skf, y_train_skf)
    
    MODEL_2_pred_skf = MODEL_2.predict(X_test_skf)
    mse_2 = mean_squared_error(y_test_skf, MODEL_2_pred_skf)
    rmse_2 = np.sqrt(mse_2)
    mae_2 = mean_absolute_error(y_test_skf, MODEL_2_pred_skf)
    r2_2 = r2_score(y_test_skf, MODEL_2_pred_skf)
    
    y_train_pred_2 = MODEL_2.predict(X_train_skf)
    train_mse_2 = mean_squared_error(y_train_skf, y_train_pred_2)
    train_r2_2 = r2_score(y_train_skf, y_train_pred_2)
    
    mmmnnn += 1
    print(' ')
    print(' ')
    print('=========================')
    print(f'{mmmnnn}')
    print('TRAINING DATA:')
    print('MSE:', train_mse_2)
    print('R2:', np.round(train_r2_2*100, 3))
    print('=========================')
    print('TESTING DATA:')
    print('MSE:', mse_2)
    print('R2:', np.round(r2_2*100, 3))




today = datetime.date.today()
formatted_today = today.strftime('%Y%m%d')

# dump(MODEL_, f'D:/2310011_Liao/水資源/done_model/drought/waterAI_{MODEL_.__class__.__name__}_new.joblib')


# print(' ')
#print('===================================')
#end_time = time.time()
#print('takes time:', (end_time - start_time)/60)






















'''

# =============================================================================================================


######## Classification model ########

#model_C = RandomForestClassifier(n_estimators = 20,
#                             criterion = 'gini',
#                             max_depth = None,
#                             min_samples_split = 7,
#                             min_samples_leaf = 1,
#                             min_weight_fraction_leaf = 0.0,
#                             max_features = 'sqrt',
#                             max_leaf_nodes = None,
#                             bootstrap = True, # 隨機抽樣
#                             oob_score = False,
#                             n_jobs = 1, 
#                             random_state = None,
#                             verbose = 0,
#                             class_weight = 'balanced'  # 自動調整類別權重
#                            )
                            
                            
#model_C = DecisionTreeClassifier()                         
                          
model_C = XGBClassifier(max_depth = 6, 
                        learning_rate = 0.3, 
                        n_estimators = 100, 
                        verbosity = None, 
                        objective = 'binary:logistic', 
                        booster = None, 
                        tree_method = None, 
                        n_jobs = None, 
                        gamma = None, 
                        min_child_weight = None, 
                        max_delta_step = None, 
                        subsample = 1, 
                        colsample_bytree = 1, 
                        colsample_bylevel = 1, 
                        colsample_bynode = 1, 
                        reg_alpha = None, 
                        reg_lambda = 1, 
                        scale_pos_weight = None, 
                        base_score = None, 
                        random_state = 0, 
                        num_parallel_tree = None, 
                        monotone_constraints = None, 
                        interaction_constraints = None, 
                        importance_type = 'gain', 
                        gpu_id = None, 
                        validate_parameters = None)


# fitting
model_C.fit(X_train, y_train)

model_C_pred = model_C.predict(X_test)
print(classification_report(y_test, model_C_pred))
print('accuracy: ',accuracy_score(y_test, model_C_pred)*100, '%')
print('訓練集: ',model_C.score(X_train, y_train)*100, '%')
print('測試集: ',model_C.score(X_test, y_test)*100, '%')
print(' ')
print(confusion_matrix(y_test, model_C_pred))

importances = model_C.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.ylabel(' ')
plt.grid(linestyle='dashed')
plt.title('Feature Importances in TREE classification [Drought prediction]')
plt.savefig('/home/ubuntu/model_data/Classification_importance.png', dpi=1000)


# Cross-validation scores
# Set up cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_score = []
for train_index, test_index in skf.split(X, y):
    X_train_skf = X.iloc[train_index, :]
    X_test_skf = X.iloc[test_index, :]
    y_train_skf = y.iloc[train_index]
    y_test_skf = y.iloc[test_index]

#    model_C_2 = RandomForestClassifier(n_estimators = 20,
#                             criterion = 'gini',
#                             max_depth = None,
#                             min_samples_split = 7,
#                             min_samples_leaf = 1,
#                             min_weight_fraction_leaf = 0.0,
#                             max_features = 'sqrt',
#                             max_leaf_nodes = None,
#                             bootstrap = True, # 隨機抽樣
#                             oob_score = False,
#                             n_jobs = 1, 
#                             random_state = None,
#                             verbose = 0,
#                             class_weight = 'balanced'  # 自動調整類別權重
#                            )


#    model_C_2 = DecisionTreeClassifier()
    
    model_C_2 = XGBClassifier(max_depth = 6, 
                        learning_rate = 0.3, 
                        n_estimators = 100, 
                        verbosity = None, 
                        objective = 'binary:logistic', 
                        booster = None, 
                        tree_method = None, 
                        n_jobs = None, 
                        gamma = None, 
                        min_child_weight = None, 
                        max_delta_step = None, 
                        subsample = 1, 
                        colsample_bytree = 1, 
                        colsample_bylevel = 1, 
                        colsample_bynode = 1, 
                        reg_alpha = None, 
                        reg_lambda = 1, 
                        scale_pos_weight = None, 
                        base_score = None, 
                        random_state = 0, 
                        num_parallel_tree = None, 
                        monotone_constraints = None, 
                        interaction_constraints = None, 
                        importance_type = 'gain', 
                        gpu_id = None, 
                        validate_parameters = None)
    
    
    
    
    model_C_2.fit(X_train_skf, y_train_skf)

    model_C_2_pred_skf = model_C_2.predict(X_test_skf)

    acc_score.append(accuracy_score(y_test_skf, model_C_2_pred_skf)) 
    print(' ')
    print(' ')
    print(f'StratifiedKFold Accuracy: {accuracy_score(y_test_skf, model_C_2_pred_skf)*100} %')



today = datetime.date.today()
formatted_today = today.strftime('%Y%m%d')
dump(model_C, '/home/ubuntu/model_data/waterAI_XGB_Classification.joblib')


'''



'''


# =============================================================================================================


######## Classification model ########

#model_C = RandomForestClassifier(n_estimators = 20,
#                             criterion = 'gini',
#                             max_depth = None,
#                             min_samples_split = 7,
#                             min_samples_leaf = 1,
#                             min_weight_fraction_leaf = 0.0,
#                             max_features = 'sqrt',
#                             max_leaf_nodes = None,
#                             bootstrap = True, # 隨機抽樣
#                             oob_score = False,
#                             n_jobs = 1, 
#                             random_state = None,
#                             verbose = 0,
#                             class_weight = 'balanced'  # 自動調整類別權重
#                            )
                            
                            
#model_C = DecisionTreeClassifier()                         
                          
model_C = XGBClassifier(max_depth = 6, 
                        learning_rate = 0.3, 
                        n_estimators = 100, 
                        verbosity = None, 
                        objective = 'binary:logistic', 
                        booster = None, 
                        tree_method = None, 
                        n_jobs = None, 
                        gamma = None, 
                        min_child_weight = None, 
                        max_delta_step = None, 
                        subsample = 1, 
                        colsample_bytree = 1, 
                        colsample_bylevel = 1, 
                        colsample_bynode = 1, 
                        reg_alpha = None, 
                        reg_lambda = 1, 
                        scale_pos_weight = None, 
                        base_score = None, 
                        random_state = 0, 
                        num_parallel_tree = None, 
                        monotone_constraints = None, 
                        interaction_constraints = None, 
                        importance_type = 'gain', 
                        gpu_id = None, 
                        validate_parameters = None)


# fitting
model_C.fit(X_train, y_train)

model_C_pred = model_C.predict(X_test)
print(classification_report(y_test, model_C_pred))
print('accuracy: ',accuracy_score(y_test, model_C_pred)*100, '%')
print('訓練集: ',model_C.score(X_train, y_train)*100, '%')
print('測試集: ',model_C.score(X_test, y_test)*100, '%')
print(' ')
print(confusion_matrix(y_test, model_C_pred))

importances = model_C.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values('Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.ylabel(' ')
plt.grid(linestyle='dashed')
plt.title('Feature Importances in TREE classification [Drought prediction]')
plt.savefig('/home/ubuntu/model_data/Classification_importance.png', dpi=1000)


# Cross-validation scores
# Set up cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_score = []
for train_index, test_index in skf.split(X, y):
    X_train_skf = X.iloc[train_index, :]
    X_test_skf = X.iloc[test_index, :]
    y_train_skf = y.iloc[train_index]
    y_test_skf = y.iloc[test_index]

#    model_C_2 = RandomForestClassifier(n_estimators = 20,
#                             criterion = 'gini',
#                             max_depth = None,
#                             min_samples_split = 7,
#                             min_samples_leaf = 1,
#                             min_weight_fraction_leaf = 0.0,
#                             max_features = 'sqrt',
#                             max_leaf_nodes = None,
#                             bootstrap = True, # 隨機抽樣
#                             oob_score = False,
#                             n_jobs = 1, 
#                             random_state = None,
#                             verbose = 0,
#                             class_weight = 'balanced'  # 自動調整類別權重
#                            )


#    model_C_2 = DecisionTreeClassifier()
    
    model_C_2 = XGBClassifier(max_depth = 6, 
                        learning_rate = 0.3, 
                        n_estimators = 100, 
                        verbosity = None, 
                        objective = 'binary:logistic', 
                        booster = None, 
                        tree_method = None, 
                        n_jobs = None, 
                        gamma = None, 
                        min_child_weight = None, 
                        max_delta_step = None, 
                        subsample = 1, 
                        colsample_bytree = 1, 
                        colsample_bylevel = 1, 
                        colsample_bynode = 1, 
                        reg_alpha = None, 
                        reg_lambda = 1, 
                        scale_pos_weight = None, 
                        base_score = None, 
                        random_state = 0, 
                        num_parallel_tree = None, 
                        monotone_constraints = None, 
                        interaction_constraints = None, 
                        importance_type = 'gain', 
                        gpu_id = None, 
                        validate_parameters = None)
    
    
    
    
    model_C_2.fit(X_train_skf, y_train_skf)

    model_C_2_pred_skf = model_C_2.predict(X_test_skf)

    acc_score.append(accuracy_score(y_test_skf, model_C_2_pred_skf)) 
    print(' ')
    print(' ')
    print(f'StratifiedKFold Accuracy: {accuracy_score(y_test_skf, model_C_2_pred_skf)*100} %')



today = datetime.date.today()
formatted_today = today.strftime('%Y%m%d')
dump(model_C, '/home/ubuntu/model_data/waterAI_XGB_Classification.joblib')


'''
