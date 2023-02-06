#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df = pd.read_csv("weather.csv")
df.head()


# In[8]:


df.tail()


# In[9]:


df[20:30]


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


#EDA-EXPLORATORY DATA ANALYSIS


# In[13]:


# SCatter plot
plt.scatter(df.MaxTemp,df.MinTemp)
plt.show()


# In[14]:


# SCatter plot
plt.scatter(df.MaxTemp,df.Rainfall)
plt.show()


# In[15]:


# creating a histogram
plt.hist(df['MaxTemp'])
plt.show()


# In[16]:


plt.scatter(df.MaxTemp,df.Evaporation)
plt.show()


# In[18]:


X= df['MinTemp'].values.reshape(-1,1)
y= df['MaxTemp'].values.reshape(-1,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)


# In[19]:


model =LinearRegression()
model.fit(X_train,y_train)


# In[20]:


print('Intercept is :',model.intercept_)


# In[21]:


print('Coefficient is :' ,model.coef_)


# In[22]:


X_train.shape 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train,y_train)
y_predict = regressor.predict( X_test)
y_predict


# In[23]:


y_pred= model.predict(X_test)
df=  pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)


# In[24]:


df1= df.head(25)
df1.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-',linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':',linewidth='0.5', color='black')
plt.show()


# In[25]:


plt.scatter(X_test,y_test,color='gray')
plt.plot(X_test,y_pred,color='red',linewidth=2)
plt.show()

print('Mean abolute error is:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared error is:', metrics.mean_squared_error(y_test,y_pred))
print('Root mean squared error is:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




