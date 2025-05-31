import numpy as np
import matplotlib.pyplot as plt
from math import *

# Alan Jociel Vazquez Torres

"""
def eulerint(F,x,y,xStop,h):
  X=[]
  Y=[]
  X.append(x)
  Y.append(y)
  while x<xStop:
    h=min(h,xStop-x)
    y=y+h*F(x,y)
    x=x+h
    X.append(x)
    Y.append(y)
  return np.array(X),np.array(Y) 
def imprimeSol(X,Y,frec):
 
  def imprimeEncabezado(n):
    print("\n x ",end=" ")
    for i in range (n):
      print(" y[",i,"] ",end=" ")
    print()

  def imprimeLinea(x,y,n):
    print("{:13.4e}".format(x),end=" ")
    for i in range (n):
      print("{:13.4e}".format(y[i]),end=" ")
    print() 
  
  m = len(Y)
  try: n = len(Y[0])
  except TypeError: n = 1
  imprimeEncabezado(n)
  for i in range(0, m, frec):
    if n == 1:
        imprimeLinea(X[i], [Y[i]], n)
    else:
        imprimeLinea(X[i], Y[i], n)

"""

"""

# Ejercicio 3, sec 7.1

def F(x,y):
    return np.sin(y)
X, Y = eulerint(F, 0, 1, 0.5, 0.1)
imprimeSol(X,Y,1)

"""

"""

# Ejercicio 4, sec 7.1

def F(x,y):
    return y**(1/3)
print('Para el caso a), y(0)=0')
X,Y=eulerint(F, 0, 0, 1, 0.1)
imprimeSol(X,Y,1)

print('Para el caso b), y(0)=10^(-16)')
X1, Y1 = eulerint(F, 0, 1e-16, 1, 0.1)
imprimeSol(X1, Y1, 1)

"""

"""

# Ejercicio 3, sec 8.1

# Metodo RK4

def Run_Kut4(F,x,y,xStop,h):
  def run_kut4(F,x,y,h):
    K0 = h*F(x,y)
    K1 = h*F(x + h/2.0, y + K0/2.0)
    K2 = h*F(x + h/2.0, y + K1/2.0)
    K3 = h*F(x + h, y + K2)
    return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
  X = []
  Y = []
  X.append(x)
  Y.append(y)
  while x < xStop:
    h = min(h,xStop - x)
    y = y + run_kut4(F,x,y,h)
    x=x+h
    X.append(x)
    Y.append(y)
  return np.array(X),np.array(Y)

def imprimeSol(X,Y,frec):
 
  def imprimeEncabezado(n):
    print("\n x ",end=" ")
    for i in range (n):
      print(" y[",i,"] ",end=" ")
    print()

  def imprimeLinea(x,y,n):
    print("{:13.4e}".format(x),end=" ")
    for i in range (n):
      print("{:13.4e}".format(y[i]),end=" ")
    print() 
  
  m = len(Y)
  try: n = len(Y[0])
  except TypeError: n = 1
  if frec == 0: frec = m
  imprimeEncabezado(n)
  for i in range(0,m,frec):
   imprimeLinea(X[i],Y[i],n)
  if i != m - 1: imprimeLinea(X[m - 1],Y[m - 1],n)

#Método de Ridder

def Ridder(f,a,b,tol=1.0e-9): 
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if np.sign(fa)!= np.sign(fb): c = a; fc = fa
  for i in range(30):
    c = 0.5*(a + b); fc = f(c)
    s = sqrt(fc**2 - fa*fb)
    if s == 0.0: return None
    dx = (c - a)*fc/s
    if (fa - fb) < 0.0: dx = -dx
    x = c + dx; fx = f(x)
    if i > 0:
      if abs(x - xOld) < tol*max(abs(x),1.0): return x
    xOld = x
    if np.sign(fc) == np.sign(fx):
      if np.sign(fa)!= np.sign(fx): b = x; fb = fx
      else: a = x; fa = fx
    else:
      a = c; b = x; fa = fc; fb = fx
  return None
  print("Demasiadas iteraciones")
  
# Inciso a)

# y'' = -exp(-y), y(0) = 1, y(1) = 0.5

def F1(x, y):
    return np.array([y[1], -np.exp(-y[0])])

def initCond_a(u):
    return np.array([1.0, u])

def r_a(u):
    X, Y = Run_Kut4(F1, 0, initCond_a(u), 1, 0.1)
    return Y[-1][0] - 0.5

u_a = Ridder(r_a, -5, 5)
print("\nEstimación de y'(0) para el caso (a):", round(u_a,3))

X_a, Y_a = Run_Kut4(F1, 0, initCond_a(u_a), 1, 0.1)
imprimeSol(X_a, Y_a, 1)

# Inciso c) 

# y'' = cos(xy), y(0) = 0, y(1) = 2

def F2(x, y):
    return np.array([y[1], np.cos(x * y[0])])

def initCond_b(u):
    return np.array([0.0, u])

def r_b(u):
    X, Y = Run_Kut4(F2, 0, initCond_b(u), 1, 0.1)
    return Y[-1][0] - 2.0

u_b = Ridder(r_b, -10, 10)
print("\nEstimación de y'(0) para el caso (b):", round(u_b,3))

X_b, Y_b = Run_Kut4(F2, 0, initCond_b(u_b), 1, 0.1)
imprimeSol(X_b, Y_b, 1)


plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)
plt.plot(X_a, [y[0] for y in Y_a], label="y(x)", color='black')
plt.title("Caso (a): y'' = -e(-y)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X_b, [y[0] for y in Y_b], label="y(x)", color='black')
plt.title("Caso (b): y'' = cos(xy)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

"""








































