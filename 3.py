import numpy as np

a=np.array([1,2,3,4,5])
b=np.array([6,7,8,9,10])

print("array a:",a)
print("Array b:",b)
print("sum a and b:",np.add(a,b))
print("sub :",np.subtract(a,b))
print("product:",np.multiply(a,b))
print("division:",np.divide(a,b))
print("exp:a",np.exp(a))
print("square root a:",np.sqrt(a))
print ("min a:",np.min(a))
print("mac of b:",np.max(b))
print("mean of a:",np.mean(a))
print("std  of b",np.std(b))
print("sum of ele:",np.sum(a))
c=np.array([[1,2],[3,4],[5,6]])
print("array c:\n",c)
print("reshaped array:")
print(np.reshape(c,(2,3)))
d=np.array([[1,2,3],[4,5,6]])

print("array d:\n",d)
print("transpose of d:")
print(np.transpose(d))
