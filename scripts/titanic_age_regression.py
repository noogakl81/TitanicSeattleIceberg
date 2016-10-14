import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


def create_age_training_test_sets(train_df, test_df):
    total_regression_data =  pd.concat([train_df,test_df])
    training_regression_data = total_regression_data[np.isnan(total_regression_data["Age"]) == False]
    test_regression_data = total_regression_data[np.isnan(total_regression_data["Age"]) == True]

    X_train = training_regression_data.drop(["Survived","Age","PassengerId"],axis=1)
    y_train = training_regression_data["Age"]
    X_test = test_regression_data.drop(["Survived","Age","PassengerId"],axis = 1)
    return X_train, y_train, X_test

def create_knn_regressor_age(X,Y,parameters):
    from sklearn.neighbors import KNeighborsRegressor

    V = np.cov(np.transpose(X))
    knn = KNeighborsRegressor(metric='mahalanobis', metric_params={'V':V})
#    knn = KNeighborsRegressor(metric='minkowski')
    sss = ShuffleSplit(n_splits=40, test_size = 0.2, random_state=472)

    grid_search_CV = GridSearchCV(knn, parameters, cv=sss)
    grid_search_CV.fit(X, Y)
    print("best score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    print grid_search_CV.grid_scores_
    knn = grid_search_CV.best_estimator_
    knn_score = knn.score(X, Y)
    print knn_score
    return knn

def fill_in_missing_age_values(train_df, test_df, clf, scaled=False, scaler=None):
    test_df["PredictedAge"] = clf.predict(test_df.drop(["Age","PassengerId"],axis = 1))
    test_df.loc[(np.isnan(test_df["Age"]) == True),"Age"] = test_df.loc[(np.isnan(test_df["Age"]) == True),"PredictedAge"]
    test_df.drop(['PredictedAge'],axis=1,inplace=True)

    train_df["PredictedAge"] = clf.predict(train_df.drop(["Age","Survived"],axis = 1))
    train_df.loc[(np.isnan(train_df["Age"]) == True),"Age"] = train_df.loc[(np.isnan(train_df["Age"]) == True),"PredictedAge"]
    train_df.drop(['PredictedAge'],axis=1,inplace=True)
    
    return train_df, test_df

def create_decision_tree_regressor_age(X,Y,parameters):
    from sklearn.tree import DecisionTreeRegressor

    regression_tree = DecisionTreeRegressor()
    regression_tree.fit(X,Y)

    from sklearn.model_selection import ShuffleSplit

    sss = ShuffleSplit(n_splits=20, test_size = 0.2, random_state=472)

    grid_search_CV = GridSearchCV(regression_tree, parameters, cv=sss)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    regression_tree = grid_search_CV.best_estimator_

    rt_score_test = regression_tree.score(X, Y)
    print "age r^2 on test data = %s" % (rt_score_test)
    return regression_tree

def fill_nan_ages_with_median_title_ages(train_df,test_df):
    median_age_mr = pd.concat([train_df[train_df['Mr']==1],test_df[test_df['Mr']==1]])['Age'].median()
    train_df.loc[(train_df['Mr']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_mr
    test_df.loc[(test_df['Mr']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_mr

    median_age_miss_young = pd.concat([train_df[train_df['Young_Miss']==1],test_df[test_df['Young_Miss']==1]])['Age'].median()
    train_df.loc[(train_df['Young_Miss']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_miss_young
    test_df.loc[(test_df['Young_Miss']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_miss_young

    median_age_miss_old = pd.concat([train_df[(train_df['Old_Miss']==1)],test_df[(test_df['Old_Miss']==1)]])['Age'].median()
    train_df.loc[(train_df['Old_Miss']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_miss_old
    test_df.loc[(test_df['Old_Miss']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_miss_old

    median_age_mrs = pd.concat([train_df[train_df['Mrs']==1],test_df[test_df['Mrs']==1]])['Age'].median()
    train_df.loc[(train_df['Mrs']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_mrs
    test_df.loc[(test_df['Mrs']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_mrs

    median_age_rev = pd.concat([train_df[train_df['Rev']==1],test_df[test_df['Rev']==1]])['Age'].median()
    train_df.loc[(train_df['Rev']==1) & np.isnan(train_df["Age"]),"Age"] = median_age_rev
    test_df.loc[(test_df['Rev']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_rev

    median_age_dr = pd.concat([train_df[train_df['Dr']==1],test_df[test_df['Dr']==1]])['Age'].median()
    train_df.loc[(train_df['Dr']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_dr
    test_df.loc[(test_df['Dr']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_dr

    median_age_mil = pd.concat([train_df[train_df['Mil']==1],test_df[test_df['Mil']==1]])['Age'].median()
    train_df.loc[(train_df['Mil']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_mil
    test_df.loc[(test_df['Mil']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_mil

    median_age_master = pd.concat([train_df[train_df['Master']==1],test_df[test_df['Master']==1]])['Age'].median()
    train_df.loc[(train_df['Master']==1) & (np.isnan(train_df["Age"])),"Age"] = median_age_master
    test_df.loc[(test_df['Master']==1) & (np.isnan(test_df["Age"])),"Age"] = median_age_master

    return train_df, test_df

def create_linear_age_regressor(X,Y,parameters):
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import Ridge

    linear_regressor = Ridge(random_state=1981)

    grid_search_CV = GridSearchCV(linear_regressor, parameters, cv=10)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    linear_regressor = grid_search_CV.best_estimator_

    print linear_regressor.coef_

    linear_regressor_score_test = linear_regressor.score(X_train_regression, Y_train_regression)

    print "age r^2 on test data = %s" % (linear_regressor_score_test)

    return linear_regressor
