import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
c = np.array([9, 10, 11, 12, 13])
d = np.array([13, 14, 15, 16, 17])

e = np.array([1, 2, 3])
f = np.array([5, 6, 7])
g = np.array([9, 10, 11])
h = np.array([13, 14, 15])
y = []
z = np.array([a, b, c, d]).T
z1 = np.array([e, f, g, h]).T
# print(z)
y.append(z)
y.append(z1)
y_new = np.concatenate(y, axis=0)
print(y_new)

bb = np.array([1, 2, 3])
aa = bb == 3

print(aa)
print(bb)
