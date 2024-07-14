import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
from data_transfomation import DataTransformation
import pandas as pd


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataInjestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")

        try:
            #Read the data as Dataframe
            df_train = pd.read_csv('notebook/data/train.csv')
            df_test = pd.read_csv('notebook/data/test.csv')
            logging.info("Reading data set")

            # Create file path for training data if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Load raw data frame into csv
            df_train.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Load training data into csv
            df_train.to_csv(self.ingestion_config.train_data_path,index=False, header=True)

            # Load test data into csv
            df_test.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of data completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__" :
    obj = DataInjestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
