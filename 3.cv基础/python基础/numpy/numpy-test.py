import numpy as np

a = np.array([1,2,3,4,3,
              3,4,5,3,1])

#idx = np.flatnonzero(a==3)

#print(idx)

a=np.reshape(a,(2,5),'C')

print(a)