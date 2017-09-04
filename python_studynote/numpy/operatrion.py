import  numpy as np
# print(np.zeros(2))
# print(np.zeros((2,2)))
# a=[[0,1,2],[2,1,3]]
# print(np.sum(a))
# #按列相加
# print(np.sum(a,axis=0))
# #按行相加
# print(np.sum(a,axis=1))
# probs = np.exp(a)
# print(probs)
# print(probs[np.newaxis:])
#print(probs /=np.sum(a))

# print(b[np.newaxis])
# ##equals c = b[np.newaxis, :]
# c = b[np.newaxis]
# print(c)
# print(b.shape)
# print(c.shape)
#看一下转置的时候有什么区别
# print(b)
# d= np.transpose(b)
# print(d)
# print(c)
# print(np.transpose(c))
# print( np.arange(7)[:None])
# print( np.arange(7)[:np.newaxis].shape)
#
#
# b = np.array([1, 2, 3],dtype=float)
# # b = np.exp(b)
# b/=np.sum(b,axis=0)
# print(b)

a = np.random.random_sample((5,5))
print(a)
print("---------------------")
print(a[0:3:1])