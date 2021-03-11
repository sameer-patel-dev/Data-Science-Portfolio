#!/usr/bin/env python
# coding: utf-8

# ### The Sparks Foundation
# **Task 1:**
# Prediction Using Supervised Machine Learning    
# **Problem Statement:**
# Predict the score of a student, if the student studies for 9.25 hours/day
# 
# **By:**
# Sameer Patel

# In[1]:


#importing all the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#importing and reaading the dataset
dataset = pd.read_csv("../Students_scores.csv")


# In[3]:


#the first five values in the dataset
dataset.head()


# In[4]:


#number of rows and columns
dataset.shape


# In[5]:


dataset.describe()


# ### Visualisation

# In[6]:


#Hours Vs Percentage of Scores
plt.scatter(dataset['Hours'], dataset['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Studied Hours')
plt.ylabel('Scores')
plt.show()


# ### Train-Test Split

# In[7]:


#X will take all the values except for the last column which is our dependent variable (target variable)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# ### Training the Simple Linear Regression model on the Training set

# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color = 'red');
plt.show()


# In[11]:


#Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)


# In[12]:


#Visualising the Training set results
plt.scatter(X_train, y_train, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[13]:


#Visualising the Test set results
plt.scatter(X_test, y_test, color = 'yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Test set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# In[14]:


#Comparing the actual values with the predicted ones.
dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
dataset 


# In[15]:


#predicting the score 
dataset = np.array(9.25)
dataset = dataset.reshape(-1, 1)
pred = regressor.predict(dataset)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# ### Error Metrics

# In[16]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# In[17]:


from sklearn.metrics import r2_score

print("The R-Square of the model is: ",r2_score(y_test,y_pred))


# ### Conclusion: 
# We used a **Linear Regression Model** to predict the score of a student if he/she studies for 9.25 hours/day and the Predicted Score came out to be **92.91**.
