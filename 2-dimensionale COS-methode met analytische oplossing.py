import numpy as np
import math
from datetime import datetime

#We bepalen de snelheid van deze code. 
start_time = datetime.now()

#Variabelen.
sigma_1=1
sigma_2=1
r=0.5
delta_1=0
delta_2=0

mu_1=r-1/2*sigma_1**2-delta_1
mu_2=r-1/2*sigma_2**2-delta_2
p=0.5

K=20
T=7/12
t_0=0

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
a=-4
b=4

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

#We definiëren eerst de functies die oplossingen van verschillende integralen representeren,
#deze hebben we nodig om de coëfficiënten H_{n_1,n_2} te bepalen.
def rho(x,u,a,c,d,K):
    if np.exp(x) > K:
        if u == c:
            rho=((4*K*u**2+K-4*u**2*np.exp(x))*math.cos(2*u*(a-x))-2*u*np.exp(x)*math.sin(2*u*(a-x)))/(4*(4*u**3+u))
        elif u == -c:
            rho=-((4*K*u**2+K-4*u**2*np.exp(x))*math.cos(2*u*(a-x))-2*u*np.exp(x)*math.sin(2*u*(a-x)))/(4*(4*u**3+u))
        else:
            rho=1/2*(-np.exp(x)*(math.sin(u*(x-a)+c*(d-x))+(c-u)*math.cos(u*(x-a)+c*(d-x)))/(c*c-2*c*u+u*u+1)+
                  np.exp(x)*(math.sin(u*(x-a)+c*(x-d))-(c+u)*math.cos(u*(x-a)+c*(x-d)))/(c*c+2*c*u+u*u+1)+
                  K*math.cos(u*(x-a)+c*(d-x))/(c-u)+K*math.cos(u*(x-a)+c*(x-d))/(c+u))
    else:
        rho=0
    return rho

def phi(x,u,a,K):
    if np.exp(x) > K:
        phi=((K*u**2+K-u**2*np.exp(x))*math.cos(u*(a-x))-u*np.exp(x)*math.sin(u*(a-x)))/(u**3+u)
    else:
        phi=0  
    return phi

def psi(x,u,a,K):
    if np.exp(x) > K:
        psi=((K*u**2+K-u**2*np.exp(x))*math.sin(u*(a-x))+u*np.exp(x)*math.cos(u*(a-x)))/(u**3+u)
    else:
        psi=0
    return psi

def f(x,u,a,K):
    if np.exp(x) > K:
        f=-1/((u**3+u)**2)*(u*(u**2*np.exp(x)*(u**2*x+x-2)-K*(u**2+1)**2*x)*math.sin(u*(a-x))+
                            (K*(u**2+1)**2-u**2*np.exp(x)*(u**2*(x+1)+x-1))*math.cos(u*(a-x)))
    else:
        f=0
    return f

def g(x,K):
    if np.exp(x) > K:
        g=np.exp(x)*(x-1)-K*x**2/2 
    else: 
        g=0 
    return g

def h(x,K):
    if np.exp(x) > K:
        h=np.exp(x)-K*x 
    else: 
        h=0 
    return h

#We definiëren hier de coëfficiënten H_{n_1,n_2} voor de call-on-max-optie,
#zie voor de afleiding ...
def H(n_1,n_2,a,b,optie_type):
    if optie_type == 'Call_On_Max':
        if n_1 == n_2 == 0:
            Vn1n2=2*(g(b,K)-g(a,K)-a*(h(b,K)-h(a,K)))
        elif n_1 == 0:
            Vn1n2=((b-a)/(n_2*np.pi)*(phi(b,n_2*np.pi/(b-a),a,K)-phi(a,n_2*np.pi/(b-a),a,K))+
                   f(b,n_2*np.pi/(b-a),a,K)-f(a,n_2*np.pi/(b-a),a,K)-
                   a*(psi(b,n_2*np.pi/(b-a),a,K)-psi(a,n_2*np.pi/(b-a),a,K)))
        elif n_2 == 0:
            Vn1n2=((b-a)/(n_1*np.pi)*(phi(b,n_1*np.pi/(b-a),a,K)-phi(a,n_1*np.pi/(b-a),a,K))+
                   f(b,n_1*np.pi/(b-a),a,K)-f(a,n_1*np.pi/(b-a),a,K)-
                   a*(psi(b,n_1*np.pi/(b-a),a,K)-psi(a,n_1*np.pi/(b-a),a,K)))
        else: 
            Vn1n2=((b-a)/(n_2*np.pi)*(rho(b,n_1*np.pi/(b-a),a,n_2*np.pi/(b-a),a,K)-
                                     rho(a,n_1*np.pi/(b-a),a,n_2*np.pi/(b-a),a,K))+
                  (b-a)/(n_1*np.pi)*(rho(b,n_2*np.pi/(b-a),a,n_1*np.pi/(b-a),a,K)-
                                     rho(a,n_2*np.pi/(b-a),a,n_1*np.pi/(b-a),a,K)))
            
    else:    
        raise ValueError("Ongeldig optietype. Gebruik 'Call_On_Max'.")
    return 4/((b-a)**2)*Vn1n2

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
N=32
print(cos(karnormal,N,'Call_On_Max'))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))