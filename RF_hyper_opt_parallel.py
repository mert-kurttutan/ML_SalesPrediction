#!/usr/bin/python3

# Python scripted for for running embarrassing paralellization with optuna
# It is just running the same script in multiple different processes
# Since study objects are connected to the same database, posgresql in this case
# they will synchronously exhaust all the parameter space
# But since this is embarrassing parallelization, it exhaust the memory very quickly
# Better make sure that you have enough RAM for the number of processes you run
# TO DO: maybe run experiments to see memory occupied by one script, multiple scripts



import pandas as pd
import numpy as np
import sklearn as sk
import gc
import matplotlib.pyplot as plt
import re
import optuna

from utils import split_time_series
from sklearn.preprocessing import MinMaxScaler


# Read data from the preprocessed data
data_sales = pd.read_pickle('data/data_sales_01.pickle')

drop_cols_fit = ["target"]
dates = data_sales['date_block_num']


test_idx = dates.max()
start_idx = 0
data_train = data_sales.loc[data_sales.date_block_num < 34, :]

data_sales.fillna(data_sales.median(), inplace=True)


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Values of the hyperparameters to be searched
search_space = {
    'n_estimators': [100],
    'max_depth': [2,3,4], # [9, 11, 13, 15, 17],
    'max_features': ['sqrt'],
    'max_samples':[0.5, 0.75], 
}


def objective_hyperparam(trial):

    # Dictionary to map from parameter name to values
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 1, 301),
        'max_depth': trial.suggest_int("max_depth", 1, 120, step=1),
        'max_features': trial.suggest_categorical("max_features", ['auto', 'sqrt']),
        'eval_metric': ['rmse'],
        'max_samples':trial.suggest_float("max_samples", 0.5, 0.9, step=0.01), 
        'seed': 42,
        'verbose' : False,
    }
    
    model = RandomForestRegressor(
                    n_estimators = params['n_estimators'],
                    max_depth = params['max_depth'],
                    max_features = params['max_features'],
                    max_samples = params['max_samples'],
                    random_state = params['seed'],
                    n_jobs=4,
                    )
    
    split_num = 3
    total_score = 0

    # Calculate score of model by taking weighted average over 5 splits
    for i in range(split_num):
        # print(10*"*" + f"This is split {i}"+10*"*")
        val_idx = dates.max() - (i+1)
        start_idx = val_idx - 12
        X_train, X_val, y_train, y_val = split_time_series(data_sales, val_idx, drop_cols_fit, start_idx)
        # print((X_train.shape, X_val.shape, y_train.shape, y_val.shape))
        
        # Apply min-max normalization
        # scaler = MinMaxScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_val = scaler.transform(X_val)

        model.fit(X_train, y_train)


        # Predictions
        yhat_val = model.predict(X_val).clip(0,20)

        # RMSE for the predictions
        rmse = mean_squared_error(yhat_val, y_val, squared=False)
        
        total_score += rmse*(split_num-i)
    
    # Take the averegge
    total_score = total_score / (split_num*(split_num+1)/2)
    
    return total_score




# optuna create-study --study-name "distributed_hyperparam_opt" --storage "postgresql://postgres:{password}@localhost/hyperparam_db"




## Posgresql commands
# DROP DATABASE IF EXISTS hyperparam_db;
# CREATE DATABASE hyperparam_db;


if __name__ == "__main__":
    
    password = "{PASSWORD}"
    
    # Study object to optimize objective_hyperparam function
    study = optuna.load_study(
        sampler=optuna.samplers.GridSampler(search_space),
        study_name="distributed_hyperparam_opt", 
        storage=f"postgresql://postgres:{password}@localhost/hyperparam_db")
    
    study.optimize(objective_hyperparam)