# model_trainer.py [ training the models ]

import os
import sys
from dataclasses import dataclass

from src.util import evaluate_model

from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.util import save_objects


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_arr, test_arr):
    
        try:
            logging.info(" Spliting Training and Test input data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1],test_arr[:,-1] 
                )
            
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "CatBoost" : CatBoostRegressor(),
                "Linear Regression" : LinearRegression(),
                "KNN" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "XGBoost" : XGBRegressor()

            }

            model_report:dict = evaluate_model(X_train = X_train, y_train= y_train, X_test = X_test, y_test = y_test, models = models)

            best_model_score = max(sorted(model_report.values())) # to get best model score

            best_model_name  = list(model_report.keys())[           # to get best model name
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(" No best model found ")

            logging.info(" Best model found")

    
            save_objects (file_path= self.model_trainer_config.trained_model_file_path , obj= best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
        
