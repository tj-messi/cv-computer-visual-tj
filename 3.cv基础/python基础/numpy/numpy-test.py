import numpy as np

a = np.array([[1,2,3,4,3],
              [3,4,5,3,1]])

#idx = np.flatnonzero(a==3)

#print(idx)

"""a=np.reshape(a,(2,5),'C')

print(a)"""

#b=np.argsort()

"""b=np.array([1,2,3])

c=a[(b)]

print(c)"""

"""c=np.sum(a,axis=0,keepdims=True)

print(c)"""

p=[1,2]
print(a[np.arange(a.shape[0]), p])