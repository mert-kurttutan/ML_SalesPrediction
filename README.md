# Machine Learning Project - Future Sales PRedicition

This repositorty contains canonical experimental ML Pipeline that consists of data preprocessing data analysis, hyperparameter tuning, training and inference. In particular, we will do all of these on <a href="https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales"> Future Sales Data </a> ensemble methods and stacking using XGBoost and Scikit.

## Description

Repository consists of the main following parts:
* EDA: Explanatory Data Analsis and Dashboard visual web app
* Feature engineering: Created and processed features, such as normalization or getting time-lagged features which is crucial future sales prediction
* Training Boosted Tree and hyperparameter tuning: Trained boosted tree model using xgboost and hyperparameter tuning using optuna
* Training of Random Forest and hyperparameter tuning: Trained Random forest model using xgboost and hyperparameter tuning using optuna,
* Stacking: Implemented stacking method based on boosted tree, random forest and linear regressor

## Getting Started

### Installing and Dependencies

To get every required package and their dependencies correct, I suggest creating new virtual env (e.g. with anaconda or miniconda). Then, run

`!git clone https://github.com/mert-kurttutan/ML_SalesPrediction.git`

`cd ML_SalesPrediction`

`pip install -r requirements.txt `

`cd data `

`unzip competitive-data-science-predict-future-sales.zip`

If you run this in enviroment with package installed already, some dependency conlfict might occur.

### Executing program

* Just run the notebook according to their number

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
