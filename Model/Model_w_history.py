import numpy as np
import pandas as pd
import pickle, gc, shap, math, random, time
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import NuSVR, SVR
from math import sqrt


def root_mean_squared_error(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def train_model_regression(X, X_test, y, params, groups, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                        'catboost_metric_name': 'MAE',
                        'sklearn_scoring_function': metrics.mean_absolute_error},
                    'mse': {'lgb_metric_name': 'mse',
                        'catboost_metric_name': 'MSE',
                        'sklearn_scoring_function': metrics.mean_squared_error},
                    'rmse': {'lgb_metric_name': 'rmse',
                        'catboost_metric_name': 'RMSE',
                        'sklearn_scoring_function': root_mean_squared_error}
                    }

    
    result_dict = {}
    
    # out-of-fold predictions on train data
    oof = np.zeros(len(X))
    
    # averaged predictions on train data
    prediction = np.zeros(len(X_test))
    
    # list of scores on folds
    scores = []
    train_loss = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    if groups is None:
        splits = folds.split(X)
        print('yes')
    else:
        splits = folds.split(X, groups = groups)
        print('no')
        
    for fold_n, (train_index, valid_index) in enumerate(splits):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X_train)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred_train = model.predict(xgb.DMatrix(X_train, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred_train = model.predict(X_train)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            train_loss.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_train, y_pred_train))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred    
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits
    
    print('Train loss mean: {0:.4f}, std: {1:.4f}.'.format(np.mean(train_loss), np.std(train_loss)))
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
        
    return result_dict


class train_config:
    
    def __init__(self, n_splits, features, date_blocks, model_type, model_params, eval_metric, 
                 early_stopping_rounds, n_estimators, seed):
        
        self.n_splits = n_splits
        self.features = features
        self.date_blocks = date_blocks
        self.model_type = model_type
        self.model_params = model_params
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.seed = seed


def model_train(train_mix, train_config):
    
    n_splits = train_config.n_splits
    seed = train_config.seed
    folds = KFold(n_splits, shuffle = True, random_state = train_config.seed)
    
    train_use = train_mix.loc[train_mix['date_block_num'].isin(train_config.date_blocks)]
    X_train = train_use.loc[(train_use.date_block_num <= 33), train_config.features]
    y_train = train_use.loc[(train_use.date_block_num <= 33), 'item_cnt_month']
    X_test = train_use.loc[(train_use.date_block_num == 34), train_config.features]
    
    result_dict = train_model_regression(
                         X=X_train, 
                         X_test=X_test, 
                         y=y_train, 
                         params=train_config.params, 
                         groups = None, 
                         folds=folds, model_type=train_config.model_type, eval_metric=train_config.eval_metric, plot_feature_importance=True,
                         verbose=500, early_stopping_rounds=train_config.early_stopping_rounds, 
                         n_estimators=train_config.n_estimators)
    
    return result_dict


if __name__ == "__main__":
    
    with open('../processed/train_mix.pickle', 'rb') as handle:
        train_mix = pickle.load(handle)
                       
    config_path = sys.argv[1]
    with open(config_path) as json_file:
        config = json.load(json_file)
        
    t_config = train_config(
        n_splits = config['n_splits'], 
        features = config['features'],
        date_blocks = config['date_blocks'],
        model_type = config['model_type'],
        model_params = config['model_params'], 
        eval_metric = config['eval_metric'],
        early_stopping_rounds = config['early_stopping_rounds'],
        n_estimators = config['n_estimators'],
        seed = config['seed'],
    )
    
    result_dict = model_train(train_mix, t_config)
    
    sample_submission = pd.read_csv('../input/sample_submission.csv')
    sample_submission['item_cnt_month'] = result_dict['prediction']
    sample_submission.to_csv(f'../results/w_history/submission_w_history_{t_config.model_type}_{t_config.seed}.csv', index=False)