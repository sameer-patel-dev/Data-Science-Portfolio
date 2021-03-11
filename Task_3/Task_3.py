#!/usr/bin/env python
# coding: utf-8

# ### The Sparks Foundation
# **Task 3:**
# Exploratory Data Analysis - Retail  
# **Problem Statement:**
# Perform Exploratory Data Analysis on dataset SampleSuperstore. As a business manager, try to find out the weak areas where you can work to make more profit. What all business problems you can derive by exploring the data?
# 
# **By:**
# Sameer Patel

# In[1]:


#importing all the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


#importing and reaading the dataset
dataset = pd.read_csv("SampleSuperstore.csv")


# In[3]:


#the first five values in the dataset
dataset.head()


# ### Dataset Cleaning

# In[4]:


#Dealing with the Missing Data in the dataset
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[5]:


#number of rows and columns
dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


dataset.head()


# In[9]:


#checking in the Category feild.
dataset.groupby(["Category"]).mean()


# In[10]:


#checking in the Category field.
dataset.groupby(["Category"]).max()


# In[11]:


#checking in the Category field.
dataset.groupby(["Category"]).min()


# In[12]:


#Finding the totalsum and visualising the comparison of total profit with respect to the sales.
plt.figure(figsize= (10,20))
dataset.groupby('Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.show()


# In[13]:


#Checking in the sub category field
dataset.groupby('Sub-Category').mean()


# In[14]:


#Checking in the sub category field
dataset.groupby('Sub-Category').max()


# In[15]:


#Checking in the sub category field
dataset.groupby('Sub-Category').min()


# In[16]:


#for the sub category
plt.figure(figsize= (10,25))
dataset.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.show()


# ### Correlation 

# In[17]:


# Correlation in Dataset
corr=dataset.corr()
corr


# In[18]:


# Heat Map Visualization
plt.figure(figsize=(5,6))
sns.heatmap(corr,annot=True,cmap='hot')


# In[19]:


#Ploting based upon the REGION and the category based.
plt.figure(figsize=(20,6))
plt.subplot(2,1,1)
sns.countplot('Region',hue='Category',data=dataset)
plt.title('REGION-CATEGORY')
plt.ylabel('COUNT',fontsize=11)
plt.xlabel('REGION',fontsize=11)
plt.grid(alpha=0.3)
plt.show()


# In[20]:


#Ploting based upon the REGION and the Sub-category based.
plt.figure(figsize=(20,6))
plt.subplot(2,1,2)
sns.countplot('Region',hue='Sub-Category',data=dataset)
plt.title('REGIONS-SUB CATEGORY')
plt.ylabel('COUNT',fontsize=11)
plt.xlabel('REGION',fontsize=11)
plt.grid(alpha=0.3)
plt.show()


# In[21]:


#Now the Analysis on the basis of shiping mode
sns.countplot(x=dataset['Ship Mode'])


# In[22]:


plt.figure(figsize=(16,12))
dataset['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# In[23]:


#Sales per Ship mode
dataset.groupby('Ship Mode')['Sales','Profit'].agg(['sum']).plot.bar()
plt.title('Year wise Total Sales & % of profit gained')


# In[24]:


#Checking fo the state wise number of delivery.
plt.figure(figsize=(13,8))
sns.countplot(x=dataset['State'])
plt.xticks(rotation=90)
plt.show()


# In[25]:


top_state=dataset.groupby('State').sum().sort_values('Profit',ascending=False)
top_state.head(15)


# In[26]:


#Pairplot
figsize=(15,10)
sns.pairplot(dataset,hue='Sub-Category')


# ### Conclusion: 
# 
# Maximum sales are from Binders, Paper, Furnishings, Phones, Storage, Art, Accessories
# Minimum sales are from Copies, Machines,and Suppliers.
# 
# Higher Numbers of Buyers are from Calofornia and New York.
# 
# Sales Loss for Texas, Pennsylvania and Ohlo.
# There is no correlation between Profit and Discount.
# 
# Profits and Sales are maximum in west region and minimum in south region.
# Profits and Sales are maximum in customer segments and minimum in Home office segments.
# Segments wise sales are almost same in every region
