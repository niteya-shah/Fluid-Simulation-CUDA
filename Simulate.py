%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun(x,y):
    return ((x**2 - y**2) * np.exp(-x**2 - y**2))

x = y = np.arange(-3.0,3.0, 0.3)
X, Y = np.meshgrid(x,y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

## %%
plt.ion()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    for j in range(len(Y)):
        ax.scatter(X[i][j], Y[i][j], Z[i][j], c='r', marker='o')
        plt.draw()
        plt.pause(0.0000000001)
## %%
%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

## %%w
plt.ion()
fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
x, y, z = [2],[3],[4]
sc = ax.scatter(x,y,z)
plt.xlim(0,10)
plt.ylim(0,10)

plt.draw()
for i in range(10):
    x.append(np.random.rand(1)*10)
    y.append(np.random.rand(1)*10)
    z.append(np.random.rand(1)*10)
    sc.set_offsets(np.c_[x,y,z])
    fig.canvas.draw_idle()
    plt.pause(0.1)

plt.waitforbuttonpress()

## %%
%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## %%
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k in range(0,100):
    x_input = np.random.rand(10,3) * 3
    ax.scatter3D(x_input.T[0], x_input.T[1], x_input.T[2])
    plt.draw()
    plt.pause(0.02)
    ax.cla()
    # y_input = np.random.rand(10) * 3
    # z_input = np.random.rand(10) * 3
## %%
