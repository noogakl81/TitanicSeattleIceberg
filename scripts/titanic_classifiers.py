import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

def create_decision_tree_classifier(X,Y,parameters):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=1981)

    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.3333, random_state=472)

    grid_search_CV = GridSearchCV(dtc, parameters, cv=sss)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    dtc = grid_search_CV.best_estimator_
    dt_score = dtc.score(X, Y)
    print "accuracy on training data = %s" % (dt_score)
    return dtc

def create_random_forest_classifier(X,Y,parameters):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)

    random_forest = RandomForestClassifier(random_state=1981)

    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.3333, random_state=472)

    grid_search_CV = GridSearchCV(random_forest, parameters, cv=sss, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    random_forest = grid_search_CV.best_estimator_

    rf_score = random_forest.score(X, Y)
    print(rf_score)
    return random_forest

def create_support_vector_machine_classifier(X,Y,parameters):
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)

    # Support Vector Machines

    svc = SVC(max_iter = 100000)

    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.33333, random_state=5195)

    grid_search_CV = GridSearchCV(svc, parameters, cv=sss, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    svc = grid_search_CV.best_estimator_

    svc_score = svc.score(X, Y)
    print(svc_score)
    return svc

def create_adaboost_classifier(X,Y,parameters):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)

    adbc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_impurity_split=4e-7,min_samples_leaf=0.075))

    grid_search_CV = GridSearchCV(adbc, parameters, cv=10, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    adbc = grid_search_CV.best_estimator_

    adbc_score = adbc.score(X, Y)
    print(adbc_score)
    return adbc

def create_gradient_boosted_classifier(X,Y,parameters):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)
    
    gbc = GradientBoostingClassifier()

    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.5, random_state=4782)

    grid_search_CV = GridSearchCV(gbc, parameters, cv=sss, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    gbc = grid_search_CV.best_estimator_

    gb_score = gbc.score(X, Y)
    print(gb_score)
    return gbc

def create_knn_classifier(X,Y,parameters):
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)

    from numpy.random import RandomState
   
    V = np.cov(np.transpose(X))

    knn = KNeighborsClassifier(metric='mahalanobis', metric_params={'V':V})
#    knn = KNeighborsClassifier(metric='minkowski',p=2)

    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.5, random_state=4782)
    grid_search_CV = GridSearchCV(knn, parameters, cv=sss, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("best score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    knn = grid_search_CV.best_estimator_

    knn_score = knn.score(X, Y)    
    print(knn_score)
    return knn

def create_logistic_regression_classifier(X,Y,parameters):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer, accuracy_score

    accuracy_scorer = make_scorer(accuracy_score)

    log_reg_clf = LogisticRegression()
    sss = StratifiedShuffleSplit(n_splits=20, test_size = 0.5, random_state=4782)
    grid_search_CV = GridSearchCV(log_reg_clf, parameters, cv=sss, scoring=accuracy_scorer)
    grid_search_CV.fit(X, Y)
    print("best score equals %f" % (grid_search_CV.best_score_))
    print("%s" % (grid_search_CV.best_params_))
    log_reg_clf = grid_search_CV.best_estimator_

    log_reg_score = log_reg_clf.score(X, Y)    
    print(log_reg_score)
    return log_reg_clf
