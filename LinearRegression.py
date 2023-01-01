import numpy as np
import matplotlib.pyplot as plt

h = np.loadtxt("h_data.txt")
t = np.loadtxt("t_data.txt")

plt.scatter(t, h)
plt.xlabel("tempo")
plt.ylabel("altura")
#plt.show()

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(t)), t, t**2])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,h)) #lado direito do sistema normal
x = np.linalg.solve(G, d)

def y(t):
    return float(x[0]) + float(x[1])*t + float(x[2])*(t**2)

plt.plot(t, y(t), 'r')
plt.show()
