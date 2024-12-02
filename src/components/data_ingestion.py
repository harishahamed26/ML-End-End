# data_ingestion.py [ ingesting the data ]

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# class for data ingestion

@dataclass # this decorator help to declare the data variables directly instead of initialising 
class DataIngestionConfig:
    train_data_path     = os.path.join("artifacts", "train.csv")
    test_data_path      = os.path.join("artifacts", "test.csv")
    raw_data_path       = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(" Entered the data ingestion ")
        
        try:
            df = pd.read_csv("notebook/data/StudentsPerformance.csv")
            logging.info(" Read the CSV data set ")
            df.rename(columns={'race/ethnicity': 'race_ethnicity', 'parental level of education': 'parental_level_of_education',
                   'test preparation course':'test_preparation_course','math score': 'math_score',
                    'reading score': 'reading_score', 'writing score':'writing_score'}, inplace=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header= True)
            
            logging.info(" Train test split initiating ")

            train, test = train_test_split(df, train_size=0.2, random_state=42)

            train.to_csv(self.data_ingestion_config.train_data_path, index = False, header= True)
            test.to_csv(self.data_ingestion_config.test_data_path, index = False, header= True)

            logging.info(" Data ingestion completed ")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
                self.data_ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()  # calling the function to initiate the data ingestion