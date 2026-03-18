import numpy as np
import scipy.special as ss

def eur_call(n, dt, S, K, u, r):
    d = 1 / u   #Step down
    p = (np.exp(r*dt)-d)/(u-d)        #Risk free probability

    returns = [0]*(n+1)
    price = 0

    for i in range(0, n+1):
        if (S*d**i*u**(n-i)>K):
            returns[i] = S*d**i*u**(n-i)-K
            price += ss.binom(n, i)*p**(n-i)*(1-p)**i*returns[i]
        else:
            returns[i] = 0


    return price

n = 4    #The number of steps in the tree
dt = 1   #The time bertween steps
S = 100  #Stock price
K = 105  #Strike price
r = 0.04 #Risk free rate
pr = 25  #The price of the call

u = 5

for i in range(-1, 49):
    if (np.exp(-r*n*dt)*eur_call(n, dt, S, K, u, r)<pr):
        u += 2**(-i)
    elif (np.exp(-r*n*dt)*eur_call(n, dt, S, K, u, r)==pr):
        break
    else:
        u -= 2**(-i)

if (u > 8.99):
    print("u is greater than 8.99")
else:
    print("The step up is ", u)