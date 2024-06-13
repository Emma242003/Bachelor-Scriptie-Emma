import numpy as np
import math
from matplotlib import pyplot as plt

#Variabelen. 
mu=0
sigma=1

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
L=8
T=1
a=-L*np.sqrt(T)
b=L*np.sqrt(T)

#Definieert de karakteristieke functie van een normale verdeling.
def karnormal(u): 
    karnormal=np.exp(1j*mu*u-sigma**2*u**2/2)
    return karnormal

#Benadert de kansdichtheidsfunctie middels cosinus Fourierreeks
def f(y,kar,N):
    F=2/(b-a)*np.real(kar(0)*np.exp(0))*math.cos(0)
    kans=1/2*F+sum([2/(b-a)*np.real(kar(n*math.pi/(b-a))*np.exp(-1j*n*a*math.pi/(b-a)))*
                        math.cos(n*math.pi*(y-a)/(b-a)) for n in range(1, N)])
    return kans

#Bepaalt waarden van de benadering van de kansdichtheidsfunctie, 
#voor verschillende waarde van N.
y = np.linspace(-5, 5, 1000)
vec_f = np.vectorize(f)  
kans = vec_f(y, karnormal, 4)
kanss = vec_f(y, karnormal, 8)    
kansss = vec_f(y, karnormal, 12)    
kanssss = vec_f(y, karnormal, 18)    

#Plot de benadering van de kansdichtheidsfunctie voor verschillende waarde van N.
plt.figure(dpi=300)

plt.plot(y, kans, color='blue', label='N=4')
plt.plot(y, kanss, color='red', label='N=8')
plt.plot(y, kansss, linestyle='dashed', color='green', label='N=12')
plt.plot(y, kanssss, color='purple', label='N=18')

plt.legend()
plt.xlabel('y')
plt.ylabel('f(y)')
