import numpy as np
import math
from datetime import datetime

#We bepalen de snelheid van deze code. 
start_time = datetime.now()

#Variabelen.
sigma=0.25
r=0.1
delta=0
mu=r-1/2*sigma**2-delta

K=120
T=0.1
t_0=0

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
L=8
a=-L*np.sqrt(T)
b=L*np.sqrt(T)

#Bepaalt X(t_0) aan de hand van S(t_0), aannemende dat X=log(S/K).
S_0=100
X_0=np.log(S_0/K)

#Definieert de karakteristieke functie van een normale verdeling.
def karnormal(u,t):
    karnormal=np.exp(1j*u*mu*t-1/2*sigma**2*u**2*t)
    #Karakteristieke functie in COS-methode is net anders, zoals in het verslag:
        #phi=e^{iuX(t_0)}*varphi.
    return np.exp(1j*u*X_0)*karnormal

#We definiëren eerst twee functies die het analytische antwoord geven van H_n.
def Chi(c,d,n):
    chi=1/(1+(n*np.pi/(b-a))**2)*(math.cos(n*np.pi*(d-a)/(b-a))*np.exp(d)- 
                                math.cos(n*np.pi*(c-a)/(b-a))*np.exp(c)+
                                n*np.pi/(b-a)*math.sin(n*np.pi*(d-a)/(b-a))*np.exp(d)-
                                n*np.pi/(b-a)*math.sin(n*np.pi*(c-a)/(b-a))*np.exp(c))
    return chi

def Lambda(c,d,n):
    if n==0:
        Lambda=d-c
    else:
        Lambda=(b-a)/(n*np.pi)*(math.sin(n*np.pi*(d-a)/(b-a))-math.sin(n*np.pi*(c-a)/(b-a)))
    return Lambda

#We definiëren hier de coëfficiënten H_n voor zowel de call- als put-optie.
def H(n,a,b,optie_type):
    if optie_type == 'call':
        Hn=2*K/(b-a)*(Chi(0,b,n)-Lambda(0,b,n))
    elif optie_type == 'put':
        Hn=2*K/(b-a)*(-Chi(a,0,n)+Lambda(a,0,n))
    else:
        raise ValueError("Ongeldig optietype. Gebruik 'call' of 'put'.")
    return Hn

#We definiëren hier de coëfficiënten F_n, in de COS-formule herkennen we
#deze coëfficiënten aan Re{phi(...)*exp(...)}.
def F(kar,n):
    return np.real(kar(n*np.pi/(b-a),T)*np.exp(-1j*n*math.pi*a/(b-a)))

#Nu gebruiken we H_n en F_n om de 1-dimensionale COS-formule te geven.
def cos(kar,N,optie_type):
    somterm=(1/2*F(kar,0)*H(0,a,b,optie_type)+
             sum([F(kar,n)*H(n,a,b,optie_type) for n in range(1,N)]))
    V=np.exp(-r*(T-t_0))*(somterm)
    return V

#Bepaalt de eerlijke prijs voor een optiecontract op tijdstip t_0.
#Hier kunnen we N ook passend kiezen.
N=156
print(cos(karnormal,N,'put'))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))