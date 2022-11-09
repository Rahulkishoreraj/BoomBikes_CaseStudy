#!/usr/bin/env python
# coding: utf-8

# ## Import Important Libaries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv("day.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


#  ## Correcting data types
# 

# In[8]:


data[["instant","season","yr","mnth","holiday","weekday","workingday","weathersit"]] = data[["instant","season","yr","mnth","holiday","weekday","workingday","weathersit"]].astype(object)
data.info()


#   ## Mapping string values to categorical columns
#   

# In[9]:


data["season"] = data["season"].replace({1:'Spring',2:"Summer",3:"Fall",4:"Winter"})
data["mnth"] = data["mnth"].replace({1:'January',2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"})
data["weekday"] = data["weekday"].replace({0:"Sunday",1:'Monday',2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"})
data["weathersit"] = data["weathersit"].replace({1:"Clear",2:"Mist",3:"Light Snow",4:"Heavy Rain"})
data.head()


# ## Visualizing the data

# plt.figure(figsize=(18,12))
# sns.pairplot(data[['temp','atemp','windspeed', 'casual','registered', 'cnt']])
# plt.show()

# ## To check correlations between numerical variables

# In[11]:



plt.figure(figsize=(18,12))
sns.heatmap(data.corr(),annot=True, cmap="cubehelix")
plt.show()


# ## Following can be seen from above:
# 1. atemp and temp are highly correlated with cnt
# 2. casual and registered are highly correlated with cnt which is expected since casual+registered= cnt

# In[12]:


#visualization of categorical variables using boxplot
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x='season', y='cnt', data = data)
plt.subplot(3,3,2)
sns.boxplot(x='yr', y='cnt', data = data)
plt.subplot(3,3,3)
sns.boxplot(x='mnth', y='cnt', data = data)
plt.subplot(3,3,4)
sns.boxplot(x='holiday', y='cnt', data = data)
plt.subplot(3,3,5)
sns.boxplot(x='weekday', y='cnt', data = data)
plt.subplot(3,3,6)
sns.boxplot(x='workingday', y='cnt', data = data)
plt.subplot(3,3,7)
sns.boxplot(x='weathersit', y='cnt', data = data)
plt.show()


# ## 1) Most bikes are rented in Fall season and in the year 2019
# ## 2) May to October period most bikes are rented
# ## 3) Most bikes were rented in mist and cloudy weather and least when light snow

# In[13]:


data.head()


# ## Data Preperation for Modelling

# In[14]:


#drop insignificant columns
data.drop(['instant', 'dteday', 'temp', 'casual', 'registered'], axis = 1, inplace = True)
data.head()


# In[15]:


#Creating dummy variables for categorical variables
data1 = pd.get_dummies(data, columns= ['season', 'mnth', 'weekday', 'weathersit' ], drop_first=True)
data1.head()


# In[16]:


#changing datatypes to numeric
data1 = data1.apply(pd.to_numeric)
data1.info()


# ## Splitting the data into Training and Testing Sets

# In[18]:


data_train, data_test = train_test_split(data1, train_size=0.7, test_size=0.3, random_state=100)
print(data_train.shape)
print(data_test.shape)


# ## Rescaling Data
# We will use MinMaxscaler

# In[19]:


scaler = MinMaxScaler()

#creating list of numeric variables
num_vars=['atemp', 'hum', 'windspeed', 'cnt']

#fit on data
data_train[num_vars]= scaler.fit_transform(data_train[num_vars])
data_train.head()


# ## Dividing data_train into x & y for model building

# In[20]:


y_train = data_train.pop('cnt')
x_train = data_train


# ## Model Building

# In[21]:


#Running RFE with output number of variables as 15

lm = LinearRegression()
lm.fit(x_train, y_train)
rfe = RFE(lm,n_features_to_select=15)
rfe = rfe.fit(x_train, y_train)

col = x_train.columns[rfe.support_]
col


# In[22]:


x_train.columns[~rfe.support_]


# ## Building model using statsmodel

# In[23]:


#creating x_test dataframe with RFE selected variables
x_train_rfe = x_train[col]
x_train_rfe.head()


# In[24]:


#Adding a constant variable
import statsmodels.api as sm

x_train_rfe = sm.add_constant(x_train_rfe)


# In[25]:


#Running the linear model
lm = sm.OLS(y_train, x_train_rfe).fit()


# In[26]:


#Summary of Linear Model

print(lm.summary())


# In[27]:


# Dropping constant

x_train_rfe = x_train_rfe.drop(['const'], axis=1)


# 
# ## Rebuilding the model after dropping weekday_Saturday

# In[28]:


x_train_new = x_train_rfe.drop(['weekday_Saturday'], axis=1)


# In[29]:



x_train_lm = sm.add_constant(x_train_new)


# In[30]:


lm = sm.OLS(y_train, x_train_lm).fit()
print(lm.summary())


# In[31]:


x_train_new.columns


# In[33]:


#checking VIF for all variables in new model

vif=pd.DataFrame()
x= x_train_new
vif['Features']= x.columns
vif['VIF']= [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif['VIF']= round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# 
# ## Rebuilding model after dropping hum

# In[34]:



x_train_new = x_train_new.drop(['hum'], axis=1)
x_train_lm = sm.add_constant(x_train_new)
lm = sm.OLS(y_train, x_train_lm).fit()
print(lm.summary())


# In[35]:


#checking VIF for all variables in new model

vif=pd.DataFrame()
x= x_train_new
vif['Features']= x.columns
vif['VIF']= [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif['VIF']= round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ## Residual Analysis

# In[37]:


#Creating y_train_pred
plt.figure(figsize=(10,6))
y_train_pred = lm.predict(x_train_lm)

#Finding residuals
res= y_train - y_train_pred

#Visualizing distribution of residuals

sns.distplot(res)
plt.xlabel("Errors")
plt.title('Error Terms', fontdict={'fontsize': 20, 'color': 'Red'})
plt.show()


# ## The residual value is centered around zero and follows Normal Distribution. Hence the model is valid.

# In[38]:


#To check if error terms are independent of each other
plt.figure(figsize=(8,6))
plt.scatter(y_train, res)
plt.show()


# ### From the above it is clear that residulas does not have sort of pattern, so the error terms are independent of each other.

# In[40]:


#Checking the if there is any linearity between variables and cnt to verify if a Liner Regression model can be used

sns.pairplot(x_train_new)
plt.show()


# ### There is no clear relationship between any of the variable , so there is no multicolinearity that exists

# ## There is no clear relationship between any of the variable , so there is no multicolinearity that exists

# In[41]:


num_vars = ['atemp', 'hum', 'windspeed', 'cnt']
data_test[num_vars]= scaler.transform(data_test[num_vars])
data_test.head()


# In[42]:


# dividing into x and y

y_test = data_test.pop('cnt')
x_test = data_test


# In[43]:


#Dropping variables which are not in the final model
x_test_new = x_test[x_train_new.columns]

#Adding a constant variable
x_test_new = sm.add_constant(x_test_new)

#Making predictions using Final Model
y_test_pred = lm.predict(x_test_new)


# ## Model Evaluation

# In[44]:


#Visualizing y_test and y_test_pred
fig= plt.figure()
fig.set_size_inches(18.5, 6.5, forward=True)
ax = fig.add_subplot(111, aspect='equal')
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_test_pred', fontsize=16)


# In[45]:


# R-squared value for final model on test set

r2=r2_score(y_test,y_test_pred)
print(r2)


# In[47]:


# Adjusted R-squared value for final model on test set

Adj_r2= 1-(1-r2)*((data_test.shape[0]-1)/(data_test.shape[0]-10-1))
print(Adj_r2)


# R-squared is 83.6% in train set and 81.8% on test set
# Adjusted R-squared is 83.2% in train set and 80.9% in test set
# These values are acceptable and hence we can say that the model is best!
# Inferences obtained from above:
# 1. Count of total rental bikes (cnt) in year 2019 is 23.5% higher than that in 2018
# 2. Cnt is 8.8% lower during the holidays
# 3. Unit increase in feeling temperature increases cnt by 41.2%
# 4. Unit increase in windspeed decreases cnt by 14.2%
# 5. Cnt is 10.9% lower in Spring
# 6. Cnt is 5.8% higher in Winter season
# 7. Cnt is 5.3% lower in the month of December
# 8. Cnt is 5.6% higher in the month of January
# 9. Cnt is 5.9% lower in the month of July
# 10. Cnt is 5.0% lower in the month of November
# 11. Cnt is 5.5% higher in month of September
# 12. Cnt is 29.1% lower when there is light snow or rain
# 13. Cnt is 8.2% lower when the weather is misty and cloudy

# In[ ]:




