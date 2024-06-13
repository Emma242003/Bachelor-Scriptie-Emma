import numpy as np
import math
from matplotlib import pyplot as plt

#Variabelen.
sigma_1=1
sigma_2=1
r=0.5
delta_1=0
delta_2=0

mu_1=r-1/2*sigma_1**2-delta_1
mu_2=r-1/2*sigma_2**2-delta_2
p=0.5

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
a=-10
b=10

#Definieert de karakteristieke functie van een bivariate normale verdeling.
def karnormal(u_1,u_2):
    karnormal=(np.exp(1j*u_1*mu_1-1/2*sigma_1**2*u_1**2+
                1j*u_2*mu_2-1/2*sigma_2**2*u_2**2-
                p*u_1*u_2*sigma_1*sigma_2))
    return karnormal

#Definieert de coëfficiënten F_{n_1,n_2}.
def F(kar, n_1, n_2):
    return 1/2*4/((b-a)**2)*(np.real(kar(n_1*np.pi/(b-a),n_2*np.pi/(b-a))*
                    np.exp(-1j*n_1*math.pi*a/(b-a)-1j*n_2*math.pi*a/(b-a)))+
                    np.real(kar(n_1*np.pi/(b-a),-n_2*np.pi/(b-a))*
                    np.exp(-1j*n_1*math.pi*a/(b-a)+1j*n_2*math.pi*a/(b-a))))

#Benadert de kansdichtheidsfunctie middels cosinus Fourierreeks.
def f(y_1, y_2, kar, N):
    somterm = 0
    for n_1 in range(0, N):
        for n_2 in range(0, N):
            # Voor de eerste somtermen waarbij n_1 of n_2 gelijk is aan 1, vermenigvuldig met een half.
            if n_1 == 0 and n_2 == 0:
                somterm += 0.25*F(kar,n_1,n_2)*math.cos(n_1*math.pi*(y_1-a)/(b-a))*math.cos(n_2*math.pi*(y_2-a)/(b-a))
            elif n_1 == 0 or n_2 == 0:
                somterm += 0.5*F(kar,n_1,n_2)*math.cos(n_1*math.pi*(y_1-a)/(b-a))*math.cos(n_2*math.pi*(y_2-a)/(b-a))
            else:
                somterm += F(kar,n_1,n_2)*math.cos(n_1*math.pi*(y_1-a)/(b-a))*math.cos(n_2*math.pi*(y_2-a)/(b-a))
    return somterm

#Bepaalt waarden van de benadering van de kansdichtheidsfunctie.
y1 = np.linspace(-4, 4, 50)
y2 = np.linspace(-4, 4, 50)
Y1, Y2 = np.meshgrid(y1, y2)

functiewaarden = np.zeros((50, 50))
for i, j in enumerate(y1):
    for k, l in enumerate(y2):
        functiewaarden[i, k] = f(j, l, karnormal, 64)  #Je kunt N hier instellen naar een passend getal.

#Plot een 3D-plot van de benadering van de kansdichtheidsfunctie.
plt.figure(dpi=300)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y1, Y2, functiewaarden, cmap='viridis')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('y_1')
ax.set_ylabel('y_2')
ax.set_zlabel('f(y_1,y_2)')

#Plot een contourplot van de benadering van de kansdichtheidsfunctie.
plt.figure(figsize=(10, 6), dpi=300)
contour_plot = plt.contourf(y1, y2, functiewaarden, cmap='viridis')
plt.colorbar(label='f(y1, y2)')
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.grid(True)

# Voeg contourlijnen toe
contour_lines = plt.contour(y1, y2, functiewaarden, colors='black', linewidths=2.5)

plt.show()
