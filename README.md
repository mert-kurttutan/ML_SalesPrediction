# Machine Learning Project - Future Sales PRedicition

This repositorty contains canonical experimental ML Pipeline that consists of data preprocessing data analysis, hyperparameter tuning, training and inference. In particular, we will do all of these on <a href="https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales"> Future Sales Data </a> ensemble methods and stacking using XGBoost and Scikit.

## Description

Repository consists of the main following parts:
* EDA: Explanatory Data Analsis and Dashboard visual web app
* Feature engineering: Created and processed features, such as normalization or getting time-lagged features which is crucial future sales prediction
* Training Boosted Tree and hyperparameter tuning: Trained boosted tree model using xgboost and hyperparameter tuning using optuna
* Training of Random Forest and hyperparameter tuning: Trained Random forest model using xgboost and hyperparameter tuning using optuna,
* Stacking: Implemented stacking method based on boosted tree, random forest and linear regressor
* Deployment: Model serving with FastAPI and giving prediction

## Getting Started

### Installing and Dependencies

To get every required package and their dependencies correct, I suggest creating new virtual env (e.g. with anaconda or miniconda). Then, run

```
git clone https://github.com/mert-kurttutan/ML_SalesPrediction.git
cd ML_SalesPrediction
pip install -r requirements.txt
cd data
unzip competitive-data-science-predict-future-sales.zip

```

If you run this in enviroment with package installed already, some dependency conlfict might occur.

### Executing program

* Just run the notebook according to their number

## Serving
I added serving/deployment part that accepts data on json format with FastAPI. To run the app, you need to build the docker image and run it. This can be as follows. First go to deplotment/app directory

`cd deployment`

Then build docker image from files therein,

`sudo docker build -t xgb_reg:v1 .`


Next, run docker container from this docker image

`sudo docker run --rm -p 81:80 xgb_reg:v1 `

Now, it can accept requests. For instance, we can send data with POST request to make predictions.
Go to deployment and use the following command in terminal to send POST request

```
curl -X POST http://localhost:81/predict \
  -d @./prediction_examples/batch_00.json \
  -H "Content-Type: application/json" 
  
```

**Note:** The data sent to the model is in preprocessed form as obtained in feature engineering notebook.


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
