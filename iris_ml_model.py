# Import the libraraies

import pandas as pd  # data preprocessing
# pandas is alised as pd
import numpy as np  # matahemaical computations
# numpy is alised as np
import matplotlib.pyplot as plt # visualization
# pyplot is aliased as plt

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Read the dataset
df = pd.read_csv('iris.csv')
# print(df.shape)   # (149,5)
# print(type(df))
# print(df.columns)   # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

#  Data Preprocessing

# 1) Handling Null values 
nv = df.isnull().sum()
# print(nv)
# There are no null values

# 2) Check duplicates
# print(df.duplicated().sum())
# 3 duplicates have been noted

# Remove the duplicates
df.drop_duplicates(inplace=True)
# print(df.duplicated().sum())

# Check the target variable
# print(df['label'].value_counts())  #  Versicolor - 50, Virginica - 49 , Setosa - 47

# Select the dependent and independent features
x = df.drop('label',axis=1)
y = df['label']
# print(x.shape) # (146,4)
# print(y.shape) # (146,)
# print(type(x)) # dataframe
# print(type(y)) # series

# split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
# print(x_train.shape)  # (109,4) 
# print(x_test.shape)   # (37,4)
# print(y_train.shape)  # (109,)
# print(y_test.shape)   # (37,)

# Train the ML Model
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=8)
knn = KNeighborsClassifier(n_neighbors=11)
rf = RandomForestClassifier(n_estimators=70,criterion='gini',max_depth=5,
                            min_samples_split=8)

lr.fit(x_train,y_train)
dt.fit(x_train,y_train)
knn.fit(x_train,y_train)
rf.fit(x_train,y_train)

# Save the Model - pickle
# Pickle is used to serialize the ml model - conversion of ml model into binary files

pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(dt,open('dt_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(rf,open('rf_model.pkl','wb'))

# wb - write binary

# To this file ion terminal, write the following
# python iris_ml_model.py
# To stop the server, write the following
# Ctrl + C
# To clear the screen
# cls









