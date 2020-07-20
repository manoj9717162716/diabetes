#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


data=pd.read_csv('diabetes.csv')


# In[5]:


data.head()


# In[6]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[7]:


data.shape


# In[8]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)


# In[9]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)


# In[10]:


ypred=model.predict(xtest)
yproba=model.predict_proba(xtest)[:,1]


# In[11]:


#for predicting, use r2_score
#for classification, use accuracy_score
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)


# In[12]:


from flask import Flask, render_template, request
app=Flask(__name__)
@app.route('/')
def xyz():
    return render_template("web.html")
@app.route('/detail',methods=['GET','POST'])
def abc():
    if(request.method=='POST'):
        a=int(request.form['v1'])
        b=int(request.form['v2'])
        c=int(request.form['v3'])
        d=int(request.form['v4'])
        e=int(request.form['v5'])
        f=int(request.form['v6'])
        g=int(request.form['v7'])
        h=int(request.form['v8'])
        pred=model.predict([[a,b,c,d,e,f,g,h]])
        return render_template('web.html',result=pred)
app.run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




