import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s1 = pd.Series([1,3,5,np.nan,6,8])
s2 = pd.Series(np.random.randn(4), index=['f', 'b', 'c', 'd'])
print s1[3]
print s2.index
print pd.Series(np.random.randn(5))
print pd.date_range('20130101', periods=6)
print pd.DataFrame(np.random.randn(6,4), index=pd.date_range('20130101', periods=6), columns=list('ABCD'))
df2 = pd.DataFrame({ 'A' : 1., 'B' : pd.Timestamp('20130102'), 'C' : pd.Series(1,index=list(range(3)),dtype='float32'), 'D' : np.array([3] *3,dtype='int32'),'E' : pd.Categorical(["test","train","test"]), 'F' : 'foo' })
print df2
print df2.all
print pd.Categorical(["test","train","test","train"])
print pd.Series(1,index=list(range(3)),dtype='float64')
print  np.array([3] * 3,dtype='int32')
print df2.head()
print df2.tail(1)
print df2.index
print df2.describe()
print df2.T
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')