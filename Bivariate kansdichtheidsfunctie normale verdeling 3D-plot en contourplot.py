import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datetime import datetime

#We bepalen de snelheid van deze code. 
start_time = datetime.now()

# Parameters voor de bivariate normale verdeling
mu = [0, 0]     # Gemiddelde vector
sigma_x = 1     # Standaarddeviatie voor x
sigma_y = 1   # Standaarddeviatie voor y
rho = 0.5    # Correlatiecoëfficiënt

# Covariantiematrix berekenen op basis van sigma's en rho
cov = [[sigma_x**2, rho * sigma_x * sigma_y],
       [rho * sigma_x * sigma_y, sigma_y**2]]

# Genereren van waarden voor de x- en y-assen
x = np.linspace(-4,4, 50)
y = np.linspace(-4,4, 50)
X, Y = np.meshgrid(x, y)

# Combineren van de x- en y-waarden in een grid
pos = np.dstack((X, Y))

# Berekenen van de kansdichtheidsfunctie (PDF)
rv = multivariate_normal(mu, cov)
pdf = rv.pdf(pos)


plt.figure(dpi=300)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, pdf, cmap='viridis')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('y_1')
ax.set_ylabel('y_2')
ax.set_zlabel('f(y_1,y_2)')


#Plot een contourplot van de benadering van de kansdichtheidsfunctie.
plt.figure(figsize=(10, 6), dpi=300)
contour_plot = plt.contourf(X,Y,pdf, cmap='viridis')
plt.colorbar(label='f(y1, y2)')
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.grid(True)

# Voeg contourlijnen toe
contour_lines = plt.contour(X,Y,pdf, colors='black', linewidths=2.5)

plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))