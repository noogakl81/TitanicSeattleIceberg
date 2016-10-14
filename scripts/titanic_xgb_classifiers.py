import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def create_optimum_xgb_classifier(X, Y, n_passes = 4):
    optimum_params = {
    'learning_rate':0.1,
    'n_estimators':1000,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'reg_alpha':0,
    'reg_lambda':1}
    for iteration in range(0,n_passes):
        optimum_params['n_estimators'] = 1000
        xgb_clf = create_xgb_classifier_from_params(optimum_params)
        xgb_fit(xgb_clf, X, Y)
        optimum_params['n_estimators'] = xgb_clf.n_estimators
        print "optimum number of estimators = %s" % (optimum_params['n_estimators'])
        # finding optimum max_depth and min_child_weight
        param_test1 = {
            'max_depth':range(3,10,2),
            'min_child_weight':range(1,10,2)
        }
        best_params = xgb_grid_search_cv(param_test1, xgb_clf, X, Y)
        optimum_params['max_depth'] = best_params['max_depth']
        optimum_params['min_child_weight'] = best_params['min_child_weight']
        xgb_clf = create_xgb_classifier_from_params(optimum_params)
        # finding optimum gamma
        param_test2 = {
            'gamma':[0,0.1,0.2,0.3,0.4]
        }
        best_params = xgb_grid_search_cv(param_test2, xgb_clf, X, Y)
        optimum_params['gamma'] = best_params['gamma']
        # fitting xgb_classifier again
        optimum_params['n_estimators'] = 1000
        xgb_clf = create_xgb_classifier_from_params(optimum_params)
        xgb_fit(xgb_clf, X, Y)
        optimum_params['n_estimators'] = xgb_clf.n_estimators
        print "optimum number of estimators = %s" % (optimum_params['n_estimators'])

        param_test3 = {
            'subsample':[i/10.0 for i in range(6,10)],
            'colsample_bytree':[i/10.0 for i in range(6,10)]
        }
        best_params = xgb_grid_search_cv(param_test3, xgb_clf, X, Y)
        optimum_params['subsample'] = best_params['subsample']
        optimum_params['colsample_bytree'] = best_params['colsample_bytree']
        xgb_clf = create_xgb_classifier_from_params(optimum_params)
        
        param_test4 = {
            'reg_alpha':[0, 1e-2, 0.1, 1, 100],
            'reg_lambda':[0, 1e-2, 0.1, 1, 100]
        }
        best_params = xgb_grid_search_cv(param_test4, xgb_clf, X, Y)
        optimum_params['reg_alpha'] = best_params['reg_lambda']
        optimum_params['reg_lambda'] = best_params['reg_lambda']

        xgb_clf = create_xgb_classifier_from_params(optimum_params)
        xgb_fit(xgb_clf, X, Y)
        optimum_params['n_estimators'] = xgb_clf.n_estimators
        print "optimum number of estimators = %s" % (optimum_params['n_estimators'])

    # adjust learning rate
    optimum_params['learning_rate'] = 0.01
    optimum_params['n_estimators'] = 1000
    xgb_clf = create_xgb_classifier_from_params(optimum_params)
    xgb_fit(xgb_clf, X, Y)
    xgb_fit(xgb_clf, X, Y, useTrainCV=False)

    optimum_params['n_estimators'] = xgb_clf.n_estimators
    print optimum_params

    return(xgb_clf)

def create_xgb_classifier_from_params(params):
    xgb_clf = xgb.XGBClassifier(learning_rate = params['learning_rate'], n_estimators = params['n_estimators'], max_depth = params['max_depth'], min_child_weight = params['min_child_weight'], gamma = params['gamma'], subsample = params['subsample'], colsample_bytree = params['colsample_bytree'], reg_alpha = params['reg_alpha'], reg_lambda = params['reg_lambda'],objective = 'binary:logistic', scale_pos_weight = 1, seed = 27)
    return xgb_clf

def xgb_grid_search_cv(params, xgb_clf, X, Y):
    gsearch = GridSearchCV(estimator = xgb_clf, param_grid = params, scoring='accuracy',cv=5)
    gsearch.fit(X,Y)
    print gsearch.best_params_
    print gsearch.best_score_
    return gsearch.best_params_

def xgb_fit(xgb_clf, X, Y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = xgb_clf.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_clf.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='error', early_stopping_rounds=early_stopping_rounds)
        xgb_clf.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    xgb_clf.fit(X, Y, eval_metric='error')
        
    #Predict training set:
    dtrain_predictions = xgb_clf.predict(X)
    dtrain_predprob = xgb_clf.predict_proba(X)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(Y, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob)
                    
    feat_imp = pd.Series(xgb_clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
