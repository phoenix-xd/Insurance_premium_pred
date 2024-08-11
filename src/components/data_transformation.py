from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from src.utilis import save_object
from src.logger import logging
from src.exception import CustomException

##data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')




#data ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformatin_objects(self):
      
        try:
            logging.info("Data Transformation initiated")
            # Segregating numerical and categorical variables
            categorical_cols = ['sex','smoker','region']
            numerical_cols =['age','bmi','children']
     
            # Define the custom ranking for each ordinal variable
            sex_categories = ["female",'male']
            smoker_categories = ['no','yes']
            region_categories =['southwest', 'southeast', 'northwest', 'northeast']    
            logging.info("pipeline Initiated")
            
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
                   

                 # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[sex_categories,smoker_categories,region_categories])),
                ('scaler',StandardScaler())
            ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])


            return preprocessor
        
            logging.info("pipeline completed")
        
        
        except Exception as e:

            logging.info("Error in Data Transformation")
            raise CustomException
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading of train and test data is completed")
            logging.info(f'train dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'test dataframe head: \n{test_df.head().to_string()}')

            logging.info('obtaining preprocessing objects')

            preprocessing_obj=self.get_data_transformatin_objects()
             
            target_column_name='expenses'
            drop_columns=[target_column_name]

            #features into independent and dependent feature

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            #applying the transformation

            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing obj on training and testing datasets")

            train_arr =np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                        )
            logging.info("preprocessor pickle is created and saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)

