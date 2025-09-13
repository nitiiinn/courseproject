import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Neighbors Classifier":KNeighborsRegressor(),
                "XGB Classifier":XGBRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            ## To get the best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise customException("No best model found",sys)
            
            logging.info(f'Best model found , Model name:{best_model_name},R2 score:{best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predictions=best_model.predict(X_test)
            r2_square=r2_score(y_test,predictions)
            return r2_square
        
        except Exception as e:
            raise customException(e,sys)

