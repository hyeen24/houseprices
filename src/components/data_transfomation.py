import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from exception import CustomException
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from dataclasses import dataclass
from utils import save_object

@dataclass
class DataTransomationConfig:
    preprocessor_ob_file_path=os.path.join("artifacts","processor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_trasnfomation_config=DataTransomationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """

        try:

            numerical_columns = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                                 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
                                 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
                                 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 
                                 'NumYearBuild', 'NumYearRemod', 'NumYearGarage']
            
            categorical_columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                                   'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                                   'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                                   'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                                   'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 
                                   'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'] 
            
            num_pipeline = Pipeline(steps =[
                ('imputer',SimpleImputer())
            ])

            cat_pipeline = Pipeline(steps=[
                ('OneHot Encoder', OneHotEncoder())
            ])

            column_transformer = ColumnTransformer([
                ('numerical transform', num_pipeline, numerical_columns),
                ('categorical transform', cat_pipeline, categorical_columns)

            ])

            return column_transformer
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try: 
            # Load dataframe from directory
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Get column transformer object
            preprocessing_obj= self.get_data_transformer_object()

            target_column_name = 'SalePrice'

            target_train_df = train_df[target_column_name]
            input_train_df = train_df.drop(columns=[target_column_name], axis=1)

            input_test_df = test_df

            input_feature_train = preprocessing_obj.fit_transform(input_train_df)
            input_feature_test = preprocessing_obj.fit_transform(input_test_df)

            train_arr = np.c_[
                input_feature_train, np.array(target_train_df)
            ]

            test_arr = np.c_[input_feature_test]

            save_object(
                file_path=self.data_trasnfomation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)




