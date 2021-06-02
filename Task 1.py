#!/usr/bin/env python
# coding: utf-8

# # GRIP- THE SPARKS FOUNDATION
# 
# Data Science and Business Analytics 

# ## Task 1- Prediction using Supervised ML
# Predict the percentage of an student based on the number of study hours they studied. This is a simple linear regression task as it involves just two variables.
# 
# ### Author - Siddhi Wandre

# Import required libraries

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Load dataset

# In[38]:


data = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data


# In[39]:


# Check if there any null value in the Dataset
data.isnull == True


# In[40]:


data.describe()


# There is no null value in the Dataset so, we can now visualize our Data.
# 

# In[41]:


data.plot.scatter(x='Hours',y='Scores',style='o')
plt.title("Study Hours vs Percentage",size=20)
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.show()


# Splitting the data

# In[42]:


#Defining x and y from the data
x = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

#Preparing Training and Testing Dataset
x_train, x_test, y_train, y_test = train_test_split(x ,y ,test_size=0.2)


# # Train the Algorithm

# In[43]:


lr = LinearRegression()
lr.fit(x_train,y_train)
print("Training complete!")


# # Plotting the regression line
# Simple linear equation (y=mx+c)

# In[44]:


line = lr.coef_*x + lr.intercept_
#plotting the test data
plt.scatter(x,y,c='b')
plt.plot(x,line,c='r',linewidth=2)

plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# # Making Prediction

# In[45]:


y_pred = lr.predict(x_test)
list(zip(y_test,y_pred))


# # Comparing Actual vs Predicted values

# In[46]:


data1 = pd.DataFrame({'Actual':y_test,'Predict':y_pred})
data1


# # Evaluating the model
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error and mean absolute error

# In[47]:


metrics.mean_absolute_error(y_test,y_pred)


# # Predicting for 9.25 hours

# In[48]:


hour=[9.25]
own_pr = lr.predict([hour])
print("No of hours = {}".format([hour]))
print("Predicted score = {}".format(own_pr[0]))


# Therefore according to the regression model is a student studies for 9.25 hours a day he/she is likely to score 94.5%.

# In[ ]:




