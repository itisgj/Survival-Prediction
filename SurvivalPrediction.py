#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


train = pd.read_csv(r'D:\Programming_practice\python1\titanic\train.csv')


# In[8]:


train.head()


# In[14]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[15]:


sns.set_style('whitegrid')



# In[18]:


sns.countplot(x='Survived',data=train,hue='Pclass')


# In[19]:


sns.distplot(train['Age'],bins=30)


# In[21]:


sns.histplot(train['Age'].dropna(),bins=30)


# In[23]:


sns.countplot(x='SibSp',data=train)


# In[26]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[27]:


train['Fare'].hist(bins=40)


# In[28]:


import cufflinks as cf


# In[29]:


cf.go_offline()


# In[30]:


train['Fare'].iplot(kind='hist',bins=30)


# In[31]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[34]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[35]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[46]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[44]:


train.dropna(inplace=True)


# In[45]:


train.drop('Cabin',axis=1,inplace=True)


# In[48]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[49]:


sex.head()


# In[52]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[53]:


embark.head()


# In[54]:


train = pd.concat([train,sex,embark],axis=1)


# In[55]:


train.head(2)


# In[56]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[57]:


train.head()


# In[59]:


train.drop('PassengerId',axis=1,inplace=True)


# In[60]:


train.head()


# In[61]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


logmodel = LogisticRegression()


# In[66]:


logmodel.fit(X_train,y_train)


# In[67]:


predictions = logmodel.predict(X_test)


# In[68]:


from sklearn.metrics import classification_report


# In[69]:


print(classification_report(y_test,predictions))


# In[70]:


from sklearn.metrics import confusion_matrix


# In[71]:


confusion_matrix(y_test,predictions)


# %%
