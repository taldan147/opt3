import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import numpy.linalg as LA

x = np.arange(0,5, 0.01)
n = np.size(x)
one = int(n / 5)
f = np.zeros(x.shape)
f[0:one] = 0.0 + 0.5*x[0:one]
f[one:2 * one] = 0.8 - 0.2 * np.log(x[100:200]);
f[(2*one):3*one] = 0.7 - 0.2*x[(2*one):3*one];
f[(3*one):4*one] = 0.3
f[(4*one):(5*one)] = 0.5 - 0.1*x[(4*one):(5*one)];

G = spdiags([-np.ones(n), np.ones(n)], np.array([0,1]), n-1,n).toarray()
etta = 0.1*np.random.randn(np.size(x));
y = f + etta
# plt.figure(); plt.plot(x,y, label = "noisy"); plt.plot(x,f, label = "clean"); plt.legend() ;plt.show()

# -------------------------------(a)-------------------------------------------------

# argmin ||x-y||+lambda/2||Gx|| --------- x = (I+lambda/2*G^T*G)*y

lam = 90
GTG = np.transpose(G) @ G
M = np.eye(n) + lam/2 * GTG
x_min = LA.inv(M) @ y

# plt.figure(); plt.plot(x,x_min, label = "recover"); plt.plot(x,f, label = "clean"); plt.legend() ;plt.show()


# --------------------------------------------(b)-------------------------------------------------

def IRLS( y, w, G ,  epsilon, lamda, maxIter):

    for i in range(maxIter):
        x = LA.inv(np.eye(n) + lamda/2 * (np.transpose(G) @ w @ G)) @ y
        for j in range (len(w)):
            w[j][j] = 1/ (abs(G[j] @ x)+ epsilon)

    return x

x_min = IRLS( y,  np.eye(n-1), G, 0.001, 1, 10)

plt.figure(); plt.plot(x,x_min, label = "recover"); plt.plot(x,f, label = "clean"); plt.legend() ;plt.show()
