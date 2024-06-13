import numpy as np
import math
from scipy.integrate import dblquad
from datetime import datetime

#We bepalen de snelheid van deze code. 
start_time = datetime.now()

#Variabelen.
sigma_1=0.2
sigma_2=0.3
r=0.048790
delta_1=0
delta_2=0

mu_1=r-1/2*sigma_1**2-delta_1
mu_2=r-1/2*sigma_2**2-delta_2
p=0.5

K=60
T=7/12
t_0=0

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
a=-10
b=10

#Bepaalt X_i(t_0) aan de hand van S_i(t_0), aannemende dat X_i=log(S_i).
S1_0=40
S2_0=40
X1_0=np.log(S1_0)
X2_0=np.log(S2_0)

#Definieert de karakteristieke functie van een bivariate normale verdeling.
def karnormal(u_1,u_2,t):
    karnormal=(np.exp(1j*u_1*mu_1*t-1/2*sigma_1**2*u_1**2*t+
              1j*u_2*mu_2*t-1/2*sigma_2**2*u_2**2*t-
              p*u_1*u_2*sigma_1*sigma_2*t))
    #Karakteristieke functie in COS-methode is net anders, zoals in het verslag:
        #phi=e^{iu_1X_1(t_0)+iu_2X_2(t_0)}*varphi.
    return np.exp(1j*u_1*X1_0+1j*u_2*X2_0)*karnormal

#We definiëren hier de coëfficiënten F_{n_1,n_2}.
def F(kar,n_1,n_2):
    return 1/2*2/(b-a)*2/(b-a)*(np.real(kar(n_1*np.pi/(b-a),n_2*np.pi/(b-a),T)*
                    np.exp(-1j*n_1*math.pi*a/(b-a)-1j*n_2*math.pi*a/(b-a)))+
                    np.real(kar(n_1*np.pi/(b-a),-n_2*np.pi/(b-a),T)*
                    np.exp(-1j*n_1*math.pi*a/(b-a)+1j*n_2*math.pi*a/(b-a))))

#We definiëren eerst twee functies die analytische H_{n_1,n_2} versimpelen,
#de integraal wordt geschreven zonder maximumfunctie, zie ...
def Integral_1(y_2,y_1,n_1,n_2):
    Integral_1=(np.exp(y_1)-K)*math.cos(n_1*np.pi*(y_1-a)/(b-a))*math.cos(n_2*np.pi*(y_2-a)/(b-a))
    return Integral_1

def Integral_2(y_2,y_1,n_1,n_2):
    Integral_2=(np.exp(y_2)-K)*math.cos(n_1*np.pi*(y_1-a)/(b-a))*math.cos(n_2*np.pi*(y_2-a)/(b-a))
    return Integral_2

#We definiëren hier de coëfficiënten H_{n_1,n_2} voor de call-on-max-optie.
def H(n_1,n_2,a,b,optie_type):
    if optie_type == 'Call_On_Max':
        if K <= np.exp(a):
            Hn1n2=4/((b-a)**2)*(dblquad(Integral_1,a,b,a, lambda y_1: y_1, args=(n_1,n_2))[0]+
                   dblquad(Integral_2,a,b,lambda y_1: y_1,b, args=(n_1,n_2))[0])
        if np.exp(a)<K<np.exp(b):
            Hn1n2=4/((b-a)**2)*(dblquad(Integral_1,a,b,a, lambda y_1: y_1, args=(n_1,n_2))[0]+
                   dblquad(Integral_2,a,b,lambda y_1: y_1,b, args=(n_1,n_2))[0]-
                   dblquad(Integral_1,a,math.log(K),a, lambda y_1: y_1, args=(n_1,n_2))[0]-
                   dblquad(Integral_2,a,math.log(K),lambda y_1: y_1,math.log(K), args=(n_1,n_2))[0])
        if np.exp(b) <= K:
            Hn1n2=0
    else:    
        raise ValueError("Ongeldig optietype. Gebruik 'Call_On_Max'.")
    return Hn1n2

#Nu gebruiken we H_{n_1,n_2} en F_{n_1,n_2} om de 2-dimensionale COS-formule te geven.
def cos(kar,N,optie_type):
    somterm=0
    for n_1 in range(0,N):
        for n_2 in range(0,N):
            # Voor de eerste somtermen waarbij n_1 of n_2 gelijk is aan 1, vermenigvuldig met een half.
            if n_1 == 0 and n_2 == 0:
                somterm += 0.25*F(kar,n_1,n_2)*H(n_1,n_2,a,b,optie_type)
            elif n_1 == 0 or n_2 == 0:
                somterm += 0.5*F(kar,n_1,n_2)*H(n_1,n_2,a,b,optie_type)
            else:
                somterm += F(kar,n_1,n_2)*H(n_1,n_2,a,b,optie_type)
    V=(b-a)**2/4*np.exp(-r*(T-t_0))*somterm
    return V

#Bepaalt de eerlijke prijs voor een optiecontract op tijdstip t_0.
#Hier kunnen we N ook passend kiezen.
N=64
print(cos(karnormal,N,'Call_On_Max'))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))