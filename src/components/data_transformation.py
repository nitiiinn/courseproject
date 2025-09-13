import os
import sys 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from src.exception import customException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformconfig()

    def get_data_transfer_object(self):
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','lunch','test_preparation_course','race_ethnicity','parental_level_of_education']

            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            cat_pipeline=Pipeline(steps=[   
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))])
            
            logging.info(f'Numerical columns:{numerical_columns}')
            logging.info(f'Categorical columns:{categorical_columns}')

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise customException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')

            preprocessor_obj=self.get_data_transfer_object()
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']  

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )   

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)] 

            return (train_arr,
                    test_arr,)
        
        except Exception as e:
            raise customException(e,sys)
        
