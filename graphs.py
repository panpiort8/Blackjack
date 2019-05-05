from algorithms import *
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')


# def f(h, d):


Hs = []; Ds = []; Vs = []
for (h, d, u), val in v.items():
    Hs.append(h)
    Ds.append(d)
    # Vs.append(val)

X, Y = np.meshgrid(Hs, Ds)
print(X)
# Z = f(X, Y)


# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
plt.show()