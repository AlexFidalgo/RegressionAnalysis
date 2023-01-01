import numpy as np
import matplotlib.pyplot as plt

# h = np.loadtxt("h_data.txt")
# t = np.loadtxt("t_data.txt")

# plt.scatter(t, h)
# plt.xlabel("tempo")
# plt.ylabel("altura")
# plt.show()

#x = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#y = np.asarray([4.19445, 5.75134, 5.27669, 5.3355, 6.64344, 6.74729, 8.51816, 9.74926, 9.57369])
x = np.asarray([1.0, 1.1, 1.3, 1.5, 1.9, 2.1])
y = np.asarray([1.84, 1.96, 2.21, 2.1, 1.95, 1.68])


def g(x, grau, r):
    res = 0
    for i in range(grau + 1):
        res = res + float(r[i])*(x**i)
    return res     
    
def erro(x, r):
    erro = 0
    for i in range(len(x)):
        erro = erro + (y[i] - g(x[i], len(r) - 1, r))**2
    return erro**(1/2)

#%%

print("\nPrimeiro Grau\n") # Regressão Linear

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(x)), x])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,y)) #lado direito do sistema normal
r = np.linalg.solve(G, d)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, g(x, 1, r), 'r')
plt.title("Primeiro Grau")
plt.show()

print(f"Coeficientes: {r}\nErro = {erro(x, r)}\n")


#%%

print("\nSegundo Grau\n")

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(x)), x, x**2])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,y)) #lado direito do sistema normal
r = np.linalg.solve(G, d)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, g(x, 2, r), 'r')
plt.title("Segundo Grau")
plt.show()

print(f"Coeficientes: {r}\nErro = {erro(x, r)}\n")

#%%

print("\nTerceiro Grau\n")

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(x)), x, x**2, x**3])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,y)) #lado direito do sistema normal
r = np.linalg.solve(G, d)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, g(x, 3, r), 'r')
plt.title("Terceiro Grau")
plt.show()

print(f"Coeficientes: {r}\nErro = {erro(x, r)}\n")

#%%

print("\nQuarto Grau\n")

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(x)), x, x**2, x**3, x**4])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,y)) #lado direito do sistema normal
r = np.linalg.solve(G, d)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, g(x, 4, r), 'r')
plt.title("Quarto Grau")
plt.show()

print(f"Coeficientes: {r}\nErro = {erro(x, r)}\n")

#%%

print("\nQuinto Grau\n")

# At * A * x = At * b => G * x = d
At = np.asmatrix([np.ones(len(x)), x, x**2, x**3, x**4, x**5])
A = np.transpose(At)
G = np.matmul(At, A) #matriz do sistema normal
d = np.transpose(np.matmul(At,y)) #lado direito do sistema normal
r = np.linalg.solve(G, d)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, g(x, 5, r), 'r')
plt.title("Quinto Grau")
plt.show()

print(f"Coeficientes: {r}\nErro = {erro(x, r)}\n")
