#!/bin/python3

#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#Loading the in-built dataset
house_dataset = fetch_california_housing()

#Creating a data frame using pandas
house_price_dataframe = pd.DataFrame(data=house_dataset.data,columns=house_dataset.feature_names)

#creating a new column 'price'
house_price_dataframe['price'] = house_dataset.target

#Finding correlation between each columns, this helps in understanding the relation between each column
correlation = house_price_dataframe.corr()

#Remove the comment if you want to see the correlation heatmap
#plt.figure(figsize=(10,10))
#sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8}, cmap='Blues')
#plt.show()

#Spliting the dataset and target
X = house_price_dataframe.drop(['price'],axis=1) #Removing the 'price' column from the dataframe and storing it in X
Y = house_price_dataframe['price'] #Creating a separate frame for target 

#print(X)
#print(Y)

#Spliting the dataframe into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1) #Use test_size=0.1 or test_size=0.2

#Displays the number of rows and columns
#print(X.shape)
#print(x_train.shape)
#print(x_test.shape)

#Loading the model
model = XGBRegressor()

#Training the model with training datasets
model.fit(x_train,y_train)

#Prediction on training dataset
training_data_prediction = model.predict(x_train)

#print(training_data_prediction)

#Evaluating the output

score1 = metrics.r2_score(y_train,training_data_prediction) #Calculating R Square Error
score2 = metrics.mean_absolute_error(y_train,training_data_prediction) #Calculating Mean Absolute Error

#Displaying the differences. 
#if score is low, the module performed well
print(f"R Square Error : {score1}")
print(f"Mean Absolute Error : {score2}")

#Visualizing the differences between the model prediction and actual value
#plt.scatter(y_train,training_data_prediction)
#plt.xlabel("Actual price")
#plt.ylabel("Predicted price")
#plt.title("Comparing Actual and Predicted Price")
#plt.show()

#Prediction on test data
test_data_prediction = model.predict(x_test)

#Evaluating the output for test data
score1 = metrics.r2_score(y_test,test_data_prediction)
score2 = metrics.mean_absolute_error(y_test,test_data_prediction)

#Displaying the error for test data
print(f"R Square Error : {score1}")
print(f"Mean Absolute Error : {score2}")