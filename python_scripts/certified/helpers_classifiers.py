import pickle
from sklearn.metrics import roc_auc_score, brier_score_loss, make_scorer, f1_score, fbeta_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate, train_test_split
from scipy.stats import uniform
from xgboost import XGBClassifier
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd

""" Model Definition for cross_validation"""
def generate_standard_model(model_str,rd_seed):
    if model_str == "svm":
        model = SVC()
        param_distributions = {'C': 10**np.random.uniform(low=-5,high=1),
                               'gamma': 10**np.random.uniform(low=-5,high=1),
                                'degree': range(1,4),
                                'kernel': ["linear", "poly", "rbf", "sigmoid"]}
    elif model_str == "adaboost":
        stump_clf =  DecisionTreeClassifier(random_state=rd_seed)
        model = AdaBoostClassifier(base_estimator = stump_clf)
        param_distributions = {"n_estimators": list(range(1,500)),
                               "learning_rate": uniform(0.01, 1),}
    elif model_str == "rforest":
        model = RandomForestClassifier(n_estimators=1000, n_jobs=1)
        param_distributions = {'criterion': ['gini', 'gini', 'entropy'],
                               'max_depth': range(2,50),
                               'min_samples_split': uniform(loc=0, scale=0.2),
                               'min_samples_leaf': uniform(loc=0, scale=0.2),
                               'bootstrap': [True, True, False]}
    elif model_str == "xgb":
        model = XGBClassifier(silent=True, objective='binary:logistic', nthread=1,scale_pos_weight=1, base_score=0.5)
        param_distributions = {"max_depth": range(3,50),
                               "learning_rate": uniform(loc=0, scale=0.1),
                               "n_estimators": range(10, 1500),
                               "min_child_weight": range(1, 200),
                               "gamma": uniform(loc=0, scale=0.1),
                               "subsample": uniform(loc=0.7, scale=0.3),
                               "colsample_bytree": uniform(loc=0.5, scale=0.5),
                               "colsample_bylevel": uniform(loc=0.1, scale=0.9),
                               "reg_alpha": uniform(loc=0, scale=0.2),
                               "reg_lambda": uniform(loc=0.8, scale=0.2)}
    else:
        raise ValueError(" Model not among the proposed one")
    return model, param_distributions

""" Inner Cross-Validation Loop """
def cv_classify_data(X, y, rd_seed,model,param_distributions, n_splits = 5, n_repeats = 5, n_iter = 500, n_jobs = 90, verbose = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state=rd_seed)
    cv_classif = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    scorer_classif = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    model_classif = RandomizedSearchCV(model, param_distributions, n_iter=n_iter,scoring=scorer_classif, n_jobs=n_jobs, cv=cv_classif, verbose=verbose)
    model.fit(X_train, y_train)
    model_best=model.best_estimator_
    model_best.fit(X_train,y_train)
    probs = model_best.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return (model_best.best_params_, model_best.best_score_, fpr, tpr, threshold, roc_auc)

""" Outer Cross-Validation Loop """
def outer_cv_loop(X, y, model_str, frac_test = .2, nb_rep_test = 5):
    dic_res={"best_params":[],"best_score":[],"auc":[],"fpr":[],"tpr":[],"threshold":[]}
    for i in tqdm(range(1,nb_rep_test+1)):
        model, param_distributions = generate_standard_model(model_str, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=frac_test,random_state=i)
        best_params, best_score, fpr, tpr, threshold, roc_auc = cv_classify_data(X, y, i,model,param_distributions)
        dic_res["best_params"].append(best_params)
        dic_res["best_score"].append(best_score)
        dic_res["fpr"].append(fpr)
        dic_res["tpr"].append(tpr)
        dic_res["threshold"].append(threshold)
        dic_res["auc"].append(roc_auc)
    return dic_res

""" Compute performance metrics over all models for chosen data"""
def test_all_models(X, y):
    return {model_name: outer_cv_loop(X, y, model_name) for model_name in ["svm","adaboost","rforest","xgb"]}
