# data_transformation.py [ transforming the data ]

import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd

# transformation libraries
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.util import save_objects

# class for data transformation config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts" , "preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transfromation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethinicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
            
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(" Numerical pipeline completed ")

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info(" Categorical encoding completed ")

            preprocesor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns ),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            logging.info(" Column Transformer completed ")

            return preprocesor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformtion(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(" train and test read completed ")
            
            logging.info(" obtaining preprocessing object ")
            
            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numerical_column = ["writing_score", "reading_score"]


            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
            target_feature_test_df= test_df[target_column_name]

            logging.info( "Applying preprocessing object on training dataframe and testing dataframe."
                )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_objects(
                file_path=self.data_transfromation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transfromation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


