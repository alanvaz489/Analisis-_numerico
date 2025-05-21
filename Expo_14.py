import numpy as np
import matplotlib.pyplot as plt


# La ecuacion del problema es:

#       a(b-θ) - θ(θ')^2
#  θ" = ----------------
#           1 + θ^2

# Donde:
    
# θ = Posicion angular
# θ'= Omega = Velocidad angular
# θ"= Aceleracion angular


#Calcular θ y θ' en un tiempo t=0.5 s

# Funcion del problema
def F(t, z):
    theta, omega = z
    a = 100
    b = 15
    dtheta_dt = omega
    domega_dt = (a * (b - theta) - theta * omega**2) / (1 + theta**2)
    return np.array([dtheta_dt, domega_dt])

# Metodo de Euler
def eulerint(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n t ", end=" ")
        for i in range(n):
            print(f" y[{i}] ", end=" ")
        print()

    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()

    m = len(Y)
    try:
        n = len(Y[0])
    except TypeError:
        n = 1

    imprimeEncabezado(n)
    for i in range(0, m, frec):
        imprimeLinea(X[i], Y[i], n)
    imprimeLinea(X[-1], Y[-1], n)

# Condiciones iniciales
theta0 = 2 * np.pi
omega0 = 0
z0 = np.array([theta0, omega0])

# Intervalo de tiempo
t0 = 0
t_final = 0.5
paso = 0.0001

# Resolver el sistema
X, Y = eulerint(F, t0, z0, t_final, paso)

imprimeSol(X, Y, frec=350)

# Separar theta y omega para graficar
Y = np.array(Y)
theta = Y[:, 0]
omega = Y[:, 1]


# y[0] = Posicion angular (θ)
# y[1] = Omega (θ')


# Graficas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(X, theta, label="θ(t)", color='blue')
plt.xlabel("Tiempo (s)")
plt.ylabel("θ (radianes)")
plt.title("Posición angular")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(X, omega, label="ω(t)", color='red')
plt.xlabel("Tiempo (s)")
plt.ylabel("ω (rad/s)")
plt.title("Velocidad angular")
plt.grid(True)

plt.tight_layout()
plt.show()