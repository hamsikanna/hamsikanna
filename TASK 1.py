#!/usr/bin/env python
# coding: utf-8

# # GRIP @ The Sparks Foundation - Data Science and Business Analytics
# 
# Task 1: Prediction using Supervised Machine Learning
# 
# level: Beginner
# 
# Author: HAMSIGA DH
# 
# Objective:
# 
# Predict the percentage of an student based on the number of hours they study per day using Simple Linear Regression technique on the student marks dataset. Also predict the score if a student studies for 9.25 hrs/ day
# 
# 
# 
# Importing all the required libraries:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[2]:


data = pd.read_csv("http://bit.ly/w-data") #Importing data using remote link
data.head(11)


# # Feature Exploration

# In[3]:


data.shape


# In[4]:


data.info()


# From the above output, we can see that there is no missing values in the dataset.

# In[5]:


data.describe()


# From the above table, we can observe that the average study hours of the student is 5.01 hours and the average scores of the student is 51.48

# In[9]:


## Plotting Scatter plot between the response vaiable and the exploratory variable

plt.figure(dpi=100)
plt.scatter(data['Hours'],data['Scores'],color = "orange")
plt.title("Marks Vs Study Hours/day",color = "green")
plt.xlabel("No. Of Hours Study/day",color = "green")
plt.ylabel("Scores",color = "green");


#  From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# In[10]:


## Pearson Correlation

correlations = data.corr(method='pearson')
print(correlations)


# From the above plot and matrix, we can see that there is a highly positive linear correlation betweeen the 2 variables. So we can conclude that, the when the no. of study hours increases the percentage scores will also increase.
# 

# # Data Preparation

# In[11]:


## Convert the data into array type

X = data.iloc[:, :-1]       #attributes
y = data.iloc[:,1].values   #labels


# In[12]:


## SPlit data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=50)


# # Model Building

# In[13]:


## Training the algorithm

lr = LinearRegression()
lr.fit(X_train,y_train)

print("Coefficient:", lr.coef_)
print("Intercept  :", lr.intercept_)


# # Here the model equation is, Scores = 3.0397 + 9.62160 * (Hours)

# In[15]:


## Plotting the Regression Line

sns.lmplot("Hours","Scores", data)
sns.set_style("ticks");


# # Scoring

# In[16]:



y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

## Comparing Actual vs Predicted values

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})  
df


# # Model Evaluation For Train Data

# In[17]:


print ('R2 score           :', round(r2_score(y_train, y_pred_train),5))
print ('Mean Squared Error :', round(mean_squared_error(y_train, y_pred_train),5))
print ('Mean absolute Error:', round(mean_absolute_error(y_train, y_pred_train),5))


# # Model Evaluation For Test Data

# In[18]:


print ('R2 score           :', round(r2_score(y_test, y_pred_test),5))
print ('Mean Squared Error :', round(mean_squared_error(y_test, y_pred_test),5))
print ('Mean absolute Error:', round(mean_absolute_error(y_test, y_pred_test),5))


# # Model validation

# In[19]:


df1 = pd.DataFrame({'R2 score': [0.94183,0.97227],'Mean Squared Error': [32.11258,21.26505],'Mean absolute Error':[5.26526,4.15649]},index = ['Train','Test'])  
df1


# # Predicting the score if a student studies for 9.25 hrs/ day

# In[20]:


test_x = [[9.25]] # X has to be a 2-D array

print(f"Predicted Score, When a student studies {test_x[0][0]} hours per day is equal to", round(lr.predict(test_x)[0],2))


# # Conclusion:
# 1.The model equation built to predict the Scores of the students is, Scores = 3.0397 + 9.62160 * (Hours)
# 2.From the estimated co-efficients, we can conclude that, when the study hour per day increases by 1 unit then the marks increases by 9.62160 units.
# 3.Here, the R-squared value is 0.97227, which indicates that 97.2% of the variance of the dependent variable (Scores) is explained by the variance of the independent variable (Study_hours/day).
# 4.If a student studeis for 9.25 hours/day, then the predicted score is equals to 92.04
# 5.Since the mean absolute error has small difference between the train and test dataset, we can conclude that themodel has good predictive power.
