import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math


def LGobj(X, c1, c2 ,w):
    XTw = np.transpose(X) @ w
    return - 1/ len(X) * ((c1 @ vec_log(vec_sigmoid(XTw))) + c2 @ vec_log(comp_sigmoid(XTw)))

def vec_sigmoid(v):
    sig_vec = []
    for i in range(len(v)):
        sig_vec.append(1 / (1 + math.exp(-v[i])))
    return sig_vec

def comp_sigmoid(v):
    sig_vec = []
    for i in range(len(v)):
        sig_vec.append(1- (1 / (1 + math.exp(-v[i]))))
    return sig_vec

def vec_log(v):
    log_vec =[]
    for i in range(len(v)):
        log_vec.append(math.log(v[i]))
    return log_vec

def gradientLogistic(X,c1,w):
    XTw = np.transpose(X) @ w
    sig_XTw = vec_sigmoid(XTw)
    return 1/len(X) * (X @ (sig_XTw - c1))

def hessianLogistic(X, w):
    XTw = np.transpose(X) @ w
    D = vec_sigmoid(XTw) @ comp_sigmoid(XTw)
    return 1 / len(X) * (X @ D @ np.transpose(X))

def generate_c(y):
    c1_label = y[0]
    c1=[1]
    for i in range(1,len(y)):
        if y[i] == c1_label :
            c1.append(1)
        else:
            c1.append(0)

    c2 = []
    for i in range(len(y)):
        if c1[i] == 1:
            c2.append(0)
        else:
            c2.append(1)

    return c1, c2

def LG(X, y, w):
    c1,c2 = generate_c(y)
    f_w = LGobj(X, c1, c2, w)
    gradient = gradientLogistic(X, c1, w)
    hessian = hessianLogistic(X, w)
    return f_w, gradient, hessian

# ------------------------------------------------------4b--------------------------------------------------------

X = np.random.rand(10,10)
y = np.random.choice([0, 1], size=10)

