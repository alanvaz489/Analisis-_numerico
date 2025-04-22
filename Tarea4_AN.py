'''
#Seccion 4.1
#Ejercicio 11


import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return x*np.sin(x)+3*np.cos(x)-x

def df(x):
    return np.sin(x)+x*np.cos(x)-3*np.sin(x)-1

def rootsearch(f, a, b, dx):
    """Encuentra todos los intervalos donde la función cambia de signo."""
    x1 = a
    f1 = f(x1)
    roots = []
    while x1 < b:
        x2 = x1+dx
        if x2 > b: x2 = b
        f2 = f(x2)

        if np.sign(f1) != np.sign(f2):  # Cambio de signo → posible raíz
            roots.append((x1, x2))

        x1 = x2
        f1 = f2
    
    return roots

def ridder(f, a, b, tol=1.0e-3):
    """Encuentra una raíz dentro de un intervalo dado usando el método de Ridder."""
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        return None  # No hay raíz en este intervalo

    while abs(b-a) > tol:
        xm = 0.5*(a+b)
        fm = f(xm)
        s = math.sqrt(fm**2-fa*fb)
        if s == 0: return xm

        x_new = xm+(xm-a)*np.sign(fa-fb)*fm/s
        f_new = f(x_new)

        if f_new == 0:
            return x_new
        elif np.sign(fm) == np.sign(f_new):
            a, fa = x_new, f_new
        else:
            b, fb = x_new, f_new

    return 0.5*(a+b)

# Buscar raíces en el intervalo (-6, 6)
dx = 0.6  
intervalo = (-5, 2) 
#Reduje el intervalo porque en el que da el ejercicio tarda mucho
#en dar el resultado.
roots_intervals = rootsearch(f, intervalo[0], intervalo[1], dx)
roots = [ridder(f, x1, x2) for x1, x2 in roots_intervals if ridder(f, x1, x2) is not None]

print("Raíces encontradas:", roots)
'''

'''
#Ejercicio 19

#The speed v of a Saturn V rocket in vertical flight near the surface of earth
#can be approximated by
# v=u*ln(M0/(M0-m*t))-g*t
#Donde:
u = 2510
M0 = 2.8e6
m = 13.3e3
g = 9.81
v = 335

import math
# Definimos la función f(t)
def f(t):
    return u*math.log(M0/(M0-m*t))-g*t-v

# Función para encontrar un intervalo donde f(t) cambie de signo
def find_interval(step=1, max_t=210.526):
    a = 0
    fa = f(a)
    
    t = a
    while t < max_t:
        t += step
        fb = f(t)
        if fa*fb < 0:
            return a, t
        a = t
        fa = fb
    
    print("No se encontró un intervalo válido.")
    return None, None


def bisection(a, b, tol=1e-6, max_iter=100):
    if f(a)*f(b) >= 0:
        print("Se requiere que f(a) y f(b) tengan signos opuestos.")
        return None
    
    for i in range(max_iter):
        c = (a+b)/2
        fc = f(c)
        
        if abs(fc) < tol or (b-a)/2 < tol:
            return c
        
        if f(a)*fc < 0:
            b = c
        else:
            a = c
    
    print("Se alcanzó el máximo de iteraciones.")
    return c

# Encontrar el intervalo
a, b = find_interval()
if a is not None and b is not None:
    print(f"Intervalo encontrado: [{a}, {b}]")
    result_bisection = bisection(a, b)
    print(f"Solución por bisección: t ≈ {result_bisection:.6f} segundos")
else:
    print("No se pudo resolver con bisección debido a la falta de un intervalo válido.")
'''

'''
#Seccion 5.1
#Ejercicio 9

# Datos de la tabla
x_val = [0.0, 0.1, 0.2, 0.3, 0.4]
f_val = [0.000000, 0.078348, 0.138910, 0.192916, 0.244981]

# Paso h (diferencia entre puntos consecutivos)
h = x_val[1] - x_val[0]  # h = 0.1

# Diferencia finita de orden O(h^4)
f_prime_02 = (f_val[0]-8*f_val[1]+8*f_val[3]-f_val[4])/(12*h)

# Diferencia finita de orden O(h^2)
f_prime_02_simple = (f_val[3]-f_val[1])/(2*h)

print(f"f'(0.2) con O(h^4) ≈ {f_prime_02}")
print(f"f'(0.2) con O(h^2) ≈ {f_prime_02_simple}")
'''

'''
#Ejercicio 10

import numpy as np

def f(x):
    return np.sin(x)

def f_prima(x):
    return np.cos(x)

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def print_results(x, h_values):
    exact = f_prima(x)
    print(f"{'h':>10} | {'Forward':>12} | {'Central':>12}")
    print("-" * 65)
    for h in h_values:
        fwd = forward_difference(f, x, h)
        ctr = central_difference(f, x, h)
        print(f"{h:10.1e} | {fwd:12.5f} | {ctr:12.5f}")

# Parámetros
x1 = 0.8
h_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

print("Cálculo de f'(0.8) con distintas h:\n")
print_results(x1, h_values)
'''


'''
#Seccion 6.1
#Ejercicio 1

def trapezoid(f,a,b,Iold,k):
    if k == 1:Inew = (f(a) + f(b))*(b - a)/2.0
    else:
        n = 2**(k -2 ) # Number of new points
        h = (b - a)/n # Spacing of new points
        x = a + h/2.0
        sum = 0.0
        for i in range(n):
            sum = sum + f(x)
            x=x+h
            Inew = (Iold + h*sum)/2.0
    return Inew

import math

def f(x): return math.log(1+math.tan(x)) #Función a evaluar.
Iold = 0.0
for k in range(1,3): #Bucle para evaluar la integral.
    Inew = round(trapezoid(f,0.0,math.pi/4,Iold,k),6) #Round para restringir a 6 decimales
    if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
    Iold = Inew
print("Integral =",Inew) #El resultado de la integral
print("nPanels =",2**(k-1)) #El numero de divisiones que se hizo.

'''
