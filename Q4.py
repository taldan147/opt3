import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math


def LGobj(X, y ,w):
    c1, c2 = generate_c(y)
    XTw = np.transpose(X) @ w
    return (- 1/ len(w)) * ((c1 @ vec_log(vec_sigmoid(XTw))) + (c2 @ vec_log(comp_sigmoid(XTw))))

def vec_sigmoid(v):
    sig_vec = []
    for i in range(len(v)):
        sig_vec.append(1 / (1 + np.exp(-v[i])))
    return np.asarray(sig_vec)

def comp_sigmoid(v):
    sig_vec = []
    for i in range(len(v)):
        sig_vec.append(1- (1 / (1 + np.exp(-v[i]))))
    return np.asarray(sig_vec)

def vec_log(v):
    log_vec =[]
    for i in range(len(v)):
        log_vec.append(math.log(v[i]))
    return np.asarray(log_vec)

def gradientLogistic(X,y,w):
    c1, c2 = generate_c(y)
    XTw = np.transpose(X) @ w
    sig_XTw = vec_sigmoid(XTw)
    return 1/len(w) * (X @ (sig_XTw - c1))

def hessianLogistic(X, w):
    XTw = np.transpose(X) @ w
    D = vec_sigmoid(XTw)*comp_sigmoid(XTw)
    D = np.diag(D)
    return 1 / len(w) * (X @ D @ np.transpose(X))

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

    return np.asarray(c1), np.asarray(c2)

def LG(X, y, w):
    f_w = LGobj(X, y, w)
    gradient = gradientLogistic(X, y, w)
    hessian = hessianLogistic(X, w)
    return f_w, gradient, hessian

# ------------------------------------------------------4b--------------------------------------------------------

X = np.random.rand(10,10)
y = np.random.choice([0, 1], size=10)
# w = np.random.uniform(low=-5, high=5, size=(10,))
w = np.random.rand(10)

d = np.random.rand(10)
# d = d * 1/ LA.norm(d)





def test_grad(X,y,w , epsilon, d, maxIter):

    loss_with_grad = []
    loss = []
    epsilons = []
    DF = []
    DF_grad = []
    for i in range(maxIter):
        epsilons.append(epsilon)
        f_eps, grad_eps, hess_eps = LG(X, y, w + epsilon * d)
        f, grad, hess = LG(X, y, w)
        loss.append(abs(f_eps-f))
        loss_with_grad.append(abs(f_eps-f-(epsilon * d @ grad)))
        epsilon = 0.5 * epsilon
        if len(loss) >1:
            DF.append(loss[i-1] / loss[i])
            DF_grad.append((loss_with_grad[i-1]/loss_with_grad[i]))

    return loss, loss_with_grad, epsilons, DF, DF_grad

def test_hess(X,y,w , epsilon, d, maxIter):
    loss_with_hess = []
    loss = []
    epsilons = []
    DF =[]
    DF_hess = []
    for i in range(maxIter):
        epsilons.append(epsilon)
        f_eps, grad_eps, hess_eps = LG(X, y, w + epsilon * d)
        f, grad, hess = LG(X, y, w)
        loss.append(LA.norm(grad_eps-grad))
        loss_with_hess.append(LA.norm(grad_eps-grad-(hess @ (epsilon * d))))
        epsilon = 0.5 * epsilon
        if len(loss) >1:
            DF.append(loss[i-1] / loss[i])
            DF_hess.append((loss_with_hess[i-1]/loss_with_hess[i]))

    return loss, loss_with_hess, epsilons, DF, DF_hess

# def test_Jacobian(X,y,w , epsilon, d, maxIter):


loss , loss_grad, epsilons, DF, DF_grad = test_grad(X, y, w, 0.1, d, 20)
iterations = np.arange(19)
print(DF_grad)
plt.figure(); plt.plot(iterations,DF, label = "loss");  plt.plot(iterations,DF_grad, label = "loss_grad"); plt.legend() ;plt.show()

loss , loss_hess, epsilons, DF, DF_hess = test_hess(X, y, w, 0.1, d, 20)
plt.figure(); plt.plot(iterations,DF, label = "loss");  plt.plot(iterations,DF_hess, label = "loss_hess"); plt.legend() ;plt.show()

# --------------------------------------------------------c-----------------------------------------------------------

def Armijio(x,  gradx, d, a, b, c, maxiter):
    for i in range(maxiter):
        obj_a = LG(X, y, x + a*d)
        if obj_a < LG(x) + c*a*(gradx @ d):
            return a
        else:
            a = b*a
    return a

def gradient_descent(X, y, w,maxIter, a0, beta, c, epsilon):
    f_values = [LG(X, y, w)]
    for i in range(maxIter):
        grad = gradientLogistic(X,y,w)
        d = -grad
        alpha = Armijio(w, grad, d, a0,beta, c, 100)
        w = w + alpha * d
        f_values.append(LG(X,y,w))
        if LA.norm(w-f_values[i-1]) / LA.norm(w) < epsilon:
            break
        return w,