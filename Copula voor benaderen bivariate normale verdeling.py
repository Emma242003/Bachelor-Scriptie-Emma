import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfinv, erf
from datetime import datetime

#We bepalen de snelheid van deze code. 
start_time = datetime.now()

#Variabelen.
mu=0
sigma_1=0.5
sigma_2=1

p=0


#Definieert de inverse cumulatieve distributiefunctie van de standaard normale verdeling.
def phi(p):
    return np.sqrt(2)*erfinv(2*p-1)

#We definiëren de marginale distributiefuncties met bijbehorende kansdichtheden.
def F(x,sigma):
    return 1/2*(1+erf((x-mu)/(sigma*np.sqrt(2))))

def f(x,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

#Hier geven we de Gaussische copula in twee dimensies met correlatie p.
def c(u_1,u_2,p,phi):
    return 1/(np.sqrt(1-p**2))*np.exp(-1/(2*(p**2-1))*(-(phi(u_1))**2*p**2+2*p*phi(u_1)*phi(u_2)-(phi(u_2))**2*p**2))

#Geeft de benaderde functie, volgt middels Stelling van Sklar, zie hoofdstuk 5.
def h(x_1,x_2):
    return c(F(x_1,sigma_1),F(x_2,sigma_2),p,phi)*f(x_1,sigma_1)*f(x_2,sigma_2)

# Genereren van een rooster van waarden.
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[h(xi, yi) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

#Plot een 3D-plot van de benadering van de kansdichtheidsfunctie.
plt.figure(dpi=300)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('y_1')
ax.set_ylabel('y_2')
ax.set_zlabel('f(y_1,y_2)')


#Plot een contourplot van de benadering van de kansdichtheidsfunctie.
plt.figure(figsize=(10, 6), dpi=300)
contour_plot = plt.contourf(X,Y,Z, cmap='viridis')
plt.colorbar(label='f(y1, y2)')
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.grid(True)
contour_lines = plt.contour(X,Y,Z, colors='black', linewidths=2.5)

plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
