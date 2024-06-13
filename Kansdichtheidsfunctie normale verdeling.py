import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

#Variabelen. 
mu=0
sigma=1

#Bepaalt de waarden van de daadwerkelijke kansdichtheidsfunctie van de normale verdeling.
y = np.linspace(-5,5,1000)

#Plot de daadwerkelijke kansdichtheidsfunctie van de normale verdeling ter vergelijking.
plt.figure(dpi=300)

plt.plot(y, stats.norm.pdf(y, mu, sigma),color='black')

plt.xlabel('y')
plt.ylabel('f(y)')

plt.show()