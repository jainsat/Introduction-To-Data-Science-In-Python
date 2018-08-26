
# coding: utf-8

# # import pandas as pd
# s=pd.Series(['orange', 'banana', 'pea'], index=['fruit', 'fruit', 'veg'])
# s

# In[29]:

import numpy as np
arr=np.array([1,2,3])
arr


# In[30]:

for a in arr:
    print(a)


# In[31]:

a = np.array([[1, 1], [2, 2], [3, 3]])
a


# In[32]:

b=np.insert(a,1,5,axis=0)
b


# In[6]:

my_dict = {1: 'a', 2: 'b', 3: 'c'}
s = pd.Series(my_dict, name="joe")
s[[1,2]]


# In[34]:

d = pd.DataFrame(s)
d


# In[35]:

sales = [{'account': 'Jones LLC', 'Jan': 150, 'Feb': 200, 'Mar': 140},
         {'account': 'Alpha Co',  'Jan': 200, 'Feb': 210, 'Mar': 215},
         {'account': 'Blue Inc',  'Jan': 50,  'Feb': 90,  'Mar': 95 }]
x=pd.Series(sales)
x


# In[36]:

l1=[1,2,3]
l2=[[1,2,3],[4,5,6]]
s=pd.Series(l1)
s


# In[37]:

x=pd.Series(l2)


# In[38]:

x


# In[39]:

d1={'a':1,'b':2, 'c':3}
s=pd.Series(d1)
s


# In[40]:

d2={'a':4,'b':5, 'c':6}
l=[d1,d2]
s=pd.Series(l, index=['s1','s2'])
s


# In[41]:

s['s3']=d1
s


# In[42]:

s['s2']=None
s


# In[43]:

a=s.drop('s2')
a


# In[44]:

l1


# In[45]:

df1=pd.DataFrame(l1)
df1


# # Single list

# In[46]:

l1=[1,2,3]
df1=pd.DataFrame(l1)
df1


# # List of lists

# In[47]:

l2=[[1,2,3],[4,5,6]]
df1=pd.DataFrame(l2)
df1


# # Dictionary

# In[48]:

d={'a':1,'b':2,'c':3}
df1=pd.DataFrame(d, index=[0])
df1
#Error: If using all scalar values, you must pass an index


# # List of dictionaries

# In[49]:

d1={'a':1,'b':2,'c':3}
d2={'a':4,'b':5,'c':6, 'd':9}
l=[d1,d2,d1]
df1=pd.DataFrame(l)
df1


# # Renaming rows and columns

# In[50]:

df1.index=['r1','r2','r3']
df1


# In[51]:

df1.columns=['c1','c2','c3','c4']
df1


# In[52]:

df1.rename(columns={'c1':'c0'},inplace=True)
df1


# # Insertion of dictionaries in column oriented manner

# In[53]:

dic = {'a':[1,2,3], 'b':[4,5,6]}
df1=pd.DataFrame(dic)
df1
# If size of lists is not same then it throws error


# # Series

# # Single series

# In[54]:

#name of series become column. All indexes of series become indexes
s= pd.Series({'a':1,'b':2,'c':3}, name='joe')
df=pd.DataFrame(s)
df


# # List of series

# In[55]:

# indexes become columns. series name become row name
s1= pd.Series({'a':1,'b':2,'c':3}, name='joe')
s2=pd.Series({'a':4,'b':5,'c':6}, name='john')
slist=[s1,s2]
df=pd.DataFrame(slist)
df


# # numpy array 

# In[56]:

import numpy as np
arr=np.array([[1,2,3],[4,5,6]])
df=pd.DataFrame(arr)
df


# # Adding index in dataframe

# In[57]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df


# In[58]:

df=df.set_index('c')
df


# In[59]:

df=df.set_index([df.index, 'a'])
df


# # Inserting a row in dataframe (in form of a list) - single index

# In[60]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index('a')
print(df)
df.loc[0]=[1,1]
print(df)
df.loc[3]=6
df


# # Inserting a row in dataframe (in form of a list) - Multiple index

# In[61]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3, 'd':14}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6, 'd': 12}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist, dtype=int)
df=df.set_index(['a','b'])
df.loc[(0,0),:]=[2,3]
df


# # Inserting a row in dataframe (in form of series)

# In[74]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='s1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='s2')
slist=[s1,s2]
df=pd.DataFrame(slist)
print(df)
s=pd.Series({'b':1}, name='s3')
df=df.append(s)
print(df)


# # Inserting column in a dataframe

# In[63]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index(['a', 'b'])
print(df)
df.loc[:,'d']=[3,4]
print(df)


# # Resetting index of data frame

# In[64]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index(['a', 'b'])
print(df)
df=df.reset_index()  # By default, index gets added as column.
print(df)


# In[65]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index(['a', 'b'])
print(df)
df=df.reset_index(drop=True, level='a') # Only a is gone here.
print(df)


# # Removing a column

# In[66]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index(['a'])
print(df)
df.drop('b', axis=1, inplace=True)
df.columns.values
#del df['col'] # del works directly on column


# # Removing a row

# In[67]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2]
df=pd.DataFrame(slist)
df=df.set_index(['a'])
print(df)
df.drop([1,4], inplace=True)
df


# # Unique values of a column

# In[69]:

import pandas as pd
s1= pd.Series({'a':1,'b':2,'c':3}, name='r1')
s2=pd.Series({'a':4,'b':5,'c':6}, name='r2')
slist=[s1,s2,s1]
df=pd.DataFrame(slist)
df=df.set_index(['a'])
print(df)
df['c'].unique()


# In[2]:

import pandas as pd
import numpy as np
s1= pd.Series({'a':2,'b':1,'c':3,'t':4}, name='r1')
s2=pd.Series({'a':4,'b':0,'c':2, 't':2}, name='r2')
s3=pd.Series({'a':4,'b':5,'c':5, 't':11}, name='r2')
s4=pd.Series({'a':4,'b':5,'c':0, 't':5}, name='r2')
slist=[s1,s2,s3,s4]
df=pd.DataFrame(slist)
df=df.set_index(['a'])
df


# In[5]:

df.loc[2]


# # apply - apply function along input axis. 0 - column 1- row

# In[1]:

import pandas as pd
import numpy as np
s1= pd.Series({'a':2,'b':1,'c':3,'t':4}, name='r1')
s2=pd.Series({'a':4,'b':0,'c':2, 't':2}, name='r2')
s3=pd.Series({'a':4,'b':5,'c':5, 't':11}, name='r2')
s4=pd.Series({'a':4,'b':5,'c':0, 't':5}, name='r3')
slist=[s1,s2,s3,s4]
df=pd.DataFrame(slist)
print(df)


# # Pass each row, axis=1 : Use it when have to apply operation among several columns of a row. : indexes in result will be df indexes.

# In[26]:

s=df.apply(lambda row: sum(row[['a','b']]), axis=1)    
# sum of a and b. Here we are returning a number of each row index.
# That's why result is a series.
s


# In[37]:

def max_min(row):
    print(type(row))
    return pd.Series({'max': np.max(row),  'min':np.min(row)}) 
#Here we're returning series for each row index. That's why result is a dataframe.
s1=df.apply(max_min, axis=1)
s1


# # Pass each column: axis=0
# 
# 

# In[32]:

print(df)


# In[2]:

s=df.apply(lambda col : sum(col), axis=0)
s


# In[38]:

d=df.apply(max_min,axis=0)
d


# # pandas.core.groupby.GroupBy.apply
# Applies apply on each dataframe, produced by  groupby

# In[42]:

def foo(df, b, c):
    print(type(df))
    return sum(df[c]*df[b])

df.groupby('a').apply(foo,'b','c')


# # groupby

# In[24]:

import pandas as pd
import numpy as np
s1= pd.Series({'a':2,'b':1,'c':3,'t':4}, name='r1')
s2=pd.Series({'a':4,'b':0,'c':2, 't':2}, name='r2')
s3=pd.Series({'a':4,'b':5,'c':5, 't':11}, name='r2')
s4=pd.Series({'a':4,'b':5,'c':0, 't':5}, name='r3')
slist=[s1,s2,s3,s4]
df=pd.DataFrame(slist)
print(df)


# In[60]:

for group, frame in df.groupby('a'):
    print('Sum of b in '  + str(group) + ' =' + str(sum(frame['b'])))


# ## agg

# # grouping on column, column becomes index

# In[61]:

df.groupby('a').agg(np.min) #applies min on every column of grouped frames


# In[22]:

d=df.groupby('a').agg([min,sum])
print(d)
d.loc[2]['b']['min']


# In[64]:

df.groupby('a').agg({'b':sum, 'c':max})


# # grouping on index

# In[28]:

print(df)
print(df.groupby(level=0).agg(np.sum))
print(df.groupby(level=0)['a'].agg({'sum': np.sum})) # Here name has been given


# In[32]:

s=pd.Series([1,2,3],[4,5,6])
s.index


# In[68]:

df/df.sum()


# # replace

# In[41]:

df=pd.DataFrame({0:['A Jain', 'b jain'],
    1: ['c Jain', 'z jain']})
df


# In[47]:

df=df.replace({'z jain':'gg'})
df


# In[45]:




# In[1]:

import pandas as pd
import numpy as np
p=pd.Series([np.NaN, 1,2])
np.mean(p)


# In[3]:

"{:,}".format(1234453)


# # Series.str.extractall

# In[23]:

d={'c1':1,'c2':2,'c3':3,'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c0':9}
import pandas as pd
import numpy as np
s=pd.Series(d)
print(s)
a=np.array(s)

a=np.arange(a,start=0,step=3)
a
#a.sum(axis=1)


# In[ ]:



