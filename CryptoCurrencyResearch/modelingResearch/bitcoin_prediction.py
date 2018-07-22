
# coding: utf-8

# In[6]:

import pandas
data=pandas.read_csv("bitcoin.csv")


# In[7]:

data.head()


# In[8]:

data.shape


# In[9]:

data=data[data.Date>'2015-01-01']


# In[10]:

data.shape


# In[19]:

data.head()


# In[20]:

data2=data.copy()


# In[59]:

bitcoin=data[['Date','Close**','Volume']]
bitcoin.head()
bitcoin=bitcoin.sort_index(ascending=False)


# In[60]:

bitcoin.head()


# In[61]:

bitcoin.shape
bitcoin.columns=['Date','close','volume']
bitcoin.head()
lag2=bitcoin.close.shift(2)
lag1=bitcoin.close.shift(1)
#bitc=pandas.merge(bitcoin,lag,how='right')
bitcoin.shape


# In[62]:

bitcoin['lag1']=lag1
bitcoin['lag2']=lag2


# In[63]:

bitcoin.head()


# In[64]:

bitcoin['vlag']=bitcoin.volume.shift(1)


# In[65]:

bitcoin.head()


# In[66]:

bitcoin.drop(['volume'],axis=1,inplace=True)


# In[67]:

bitcoin.head()


# In[ ]:




# In[68]:

train,test=bitcoin[bitcoin.Date<='2017-09-01'],bitcoin[bitcoin.Date>'2017-09-01']


# In[69]:

train.shape


# In[70]:

test.shape


# In[74]:

train.drop([1240,1239],axis=0,inplace=True)


# In[75]:

train.head()


# In[72]:

test.head()


# In[77]:

train.drop(['Date'],axis=1,inplace=True)
test.drop(['Date'],axis=1,inplace=True)
train.reset_index(inplace=True)
test.reset_index(inplace=True)


# In[79]:

test.head()


# In[81]:

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit(train)
train=sc.transform(train)
test=sc.transform(test)
xtr,ytr=train[:,2:],train[:,1]
xts,yts=test[:,2:],test[:,1]


# In[83]:

train[:5,:]


# In[85]:

from keras import layers,models


# In[87]:

xtr.shape


# In[88]:

xts.shape


# In[90]:

#[sample,timestep,features]
xtr=xtr.reshape(xtr.shape[0],1,xtr.shape[1])
xts=xts.reshape(xts.shape[0],1,xts.shape[1])


# # LST Model

# In[97]:

model=models.Sequential()


# In[98]:

model.add(layers.LSTM(10,input_shape=(1,3)))


# In[93]:

model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dropout(5))

# In[99]:

model.add(layers.Dense(1))


# In[100]:
from sklearn.metrics import mean_squared_error,r2_score
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])


# In[102]:

model.fit(xtr,ytr,epochs=10,batch_size=1,verbose=True,validation_data=(xts,yts))


# In[ ]:
ip=numpy.array([1,7480,7587,7557,6049220000]).reshape(1,5)
ip=sc.transform(ip)
ip=ip[:,2:]
ip=ip.reshape(1,1,3)
out=model.predict(ip)
op=numpy.array([0,out,0,0,0]).reshape(1,5)
sc.inverse_transform(op)