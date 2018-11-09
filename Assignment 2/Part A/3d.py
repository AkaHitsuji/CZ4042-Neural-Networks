from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
acc = np.array([])
for i in range(5,101,5):
	acc2 =np.array([])
	for j in range(5,101,5):
		acc2 = np.append(acc2,j)
	acc = np.append(acc,acc2,axis=0)
acc = acc.reshape((20,20))
print(acc)


x = range(5,101,5)
y = range(5,101,5)
X,Y = np.meshgrid(x,y)
Z = np.random.rand(20,20)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, acc, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

plt.savefig('hello')