# VERSION : python 3.12.0
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Sort data by year
def sort_dataset(dataset_df):
   return dataset_df.sort_values(by='year')

# Split data as train/test
def split_dataset(dataset_df):
   dataset_df['salary'] *= 0.001
  
  # label : salary
   X_train = dataset_df.drop(columns='salary', axis=1)[:1718]
   Y_train = dataset_df['salary'][:1718]
   X_test = dataset_df.drop(columns='salary', axis=1)[1718:]
   Y_test = dataset_df['salary'][1718:]

   return X_train, X_test, Y_train, Y_test

# Extract numerical columns
def extract_numerical_cols(dataset_df):
   numerical_cols = np.array(['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war'])
   
   # Drop not numerical_cols
   drop_cols = np.setdiff1d(dataset_df.columns, numerical_cols)

	# making dataset except for not numerical_cols
   for drop_col in drop_cols:
      dataset_df = dataset_df.drop([drop_col], axis=1)

   return dataset_df

# Train and Predict by DecisionTree Using regressor
def train_predict_decision_tree(X_train, Y_train, X_test):
   dt_cls = DecisionTreeRegressor()
   dt_cls = dt_cls.fit(X_train, Y_train)

   return dt_cls.predict(X_test)

# Train and Predict by RandomForest Using regressor
def train_predict_random_forest(X_train, Y_train, X_test):
   rf_cls = RandomForestRegressor()
   rf_cls = rf_cls.fit(X_train, Y_train)

   return rf_cls.predict(X_test)

# Train and Predict by svm Using SVR, pipeline
def train_predict_svm(X_train, Y_train, X_test):
   svm_pipe = make_pipeline(
      StandardScaler(),
      SVR()
   )
   svm_pipe.fit(X_train, Y_train)

   return svm_pipe.predict(X_test)

# Calculate RMSE
def calculate_RMSE(labels, predictions):
   return np.sqrt(np.mean((predictions - labels) ** 2))

if __name__=='__main__':
   #DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
   data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
   
   sorted_df = sort_dataset(data_df)
   X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
   
   X_train = extract_numerical_cols(X_train)
   X_test = extract_numerical_cols(X_test)

   dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
   rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
   svm_predictions = train_predict_svm(X_train, Y_train, X_test)
   
   print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))   
   print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))   
   print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions)) 