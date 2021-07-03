import copy

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import struct
from array import array
from os.path import join
import os

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
        if v[i] > 0:
            log_vec.append(math.log(v[i]))
        else:
            log_vec.append(v[i])
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



loss , loss_grad, epsilons, DF, DF_grad = test_grad(X, y, w, 0.1, d, 20)
iterations = np.arange(19)
iterations2 = np.arange(20)
loss2 , loss_hess, epsilons, DF2, DF_hess = test_hess(X, y, w, 0.1, d, 20)

plt.figure(); plt.semilogy(iterations2,loss, label = "gradient test : loss");  plt.semilogy(iterations2,loss_grad, label = "loss_grad");

plt.suptitle('Tests by descent factor')
plt.xlabel('Iterations')
plt.ylabel('Decrease factor')
plt.semilogy(iterations2,loss2, label = "hessian test : loss");  plt.semilogy(iterations2,loss_hess, label = "loss_hess"); plt.legend() ;plt.show()

plt.figure(); plt.plot(iterations,DF, label = "gradient test : loss");  plt.plot(iterations,DF_grad, label = "loss_grad");
plt.suptitle(r'Tests by decreasing values of $\epsilon$')
plt.xlabel('Iterations')
plt.ylabel(r'$\epsilon$')

plt.plot(iterations,DF2, label = "hessian test : loss");  plt.plot(iterations,DF_hess, label = "loss_hess"); plt.legend() ;plt.show()


# --------------------------------------------------------c-----------------------------------------------------------
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            # meanval = np.mean(img)
            # stdval = np.std(img)
            # img = (img - meanval) / (stdval + 0.1)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

    #


# Verify Reading Dataset via MnistDataloader class
#
#
# Set file paths based on added MNIST Datasets
#
cwd = os.getcwd()
input_path = cwd + '\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1
    plt.show()


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
#
# Show some random training and test images
#
# images_2_show = []
# titles_2_show = []
# for i in range(0, 10):
#     r = random.randint(1, 60000)
#     images_2_show.append(x_train[r])
#     titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))
#
# for i in range(0, 5):
#     r = random.randint(1, 10000)
#     images_2_show.append(x_test[r])
#     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

# show_images(images_2_show, titles_2_show)


def Armijio(X,y,x, gradx, d, a, b, c, maxiter):
    for i in range(maxiter):
        obj_a = LGobj(X, y, x + a*d)
        if obj_a < (LGobj(X, y, x) + c*a*np.inner(gradx,d)):
            print(i)
            return a
        else:
            a = b*a
    return a


def gradient_descent(X, y, w,maxIter, a0, beta, c, epsilon):
    f_values = [LGobj(X, y, w)]
    for i in range(maxIter):
        grad = gradientLogistic(X,y,w)
        d = -grad
        alpha = Armijio(X,y,w, grad, d, a0,beta, c, 100)
        wPrev = w
        w = w + alpha * d
        w = np.clip(w,-1,1)
        f_values.append(LGobj(X,y,w))
        if LA.norm(w-wPrev) / LA.norm(w) < epsilon:
            break
    return w, f_values

def exactNewton(X, y, w,maxIter, a0, beta, c, epsilon):
    f_values = [LGobj(X, y, w)]
    for i in range(maxIter):
        grad = gradientLogistic(X,y,w)
        hess = hessianLogistic(X,w) + np.eye(len(w))* 0.01
        d = -1 * LA.inv(hess) @ grad
        alpha = Armijio(X,y,w, grad, d, a0,beta, c, 100)
        wPrev = w
        w = w + alpha * d
        w = np.clip(w,-1,1)
        f_values.append(LGobj(X,y,w))
        if LA.norm(w-wPrev) / LA.norm(w) < epsilon:
            break
    return w, f_values


def filterTwoDigits(num1, num2, x, y):
    x_res=[]
    y_res = []
    for i in range(len(x)):
        if y[i] == num1 or y[i] == num2:
            x_res.append(x[i])
            y_res.append(y[i])
    return x_res ,y_res

def imageToVector(data):
    xNew = copy.deepcopy(data)
    for i in range(len(xNew)):
        # tmpMatrix = xNew[i]
        # tmp = np.ndarray.tolist(tmpMatrix[0])
        # for j in range(1, len(tmpMatrix)):
        #     list = np.ndarray.tolist(tmpMatrix[j])
        #     tmp.extend(list)
        xNew[i] = np.asarray(np.ndarray.flatten(xNew[i]))
    return xNew


def calculateError(w, w_k):
    ans = []
    for i in range(len(w_k)):
        ans.append(abs(w_k[i]-w))
    return ans;

x_train = [x_train[i] for i in range(30000)]



def runAndPlotGD(digit1, digit2):
    zero_one_images_train, zero_one_labels_train = filterTwoDigits(digit1, digit2, x_train, y_train)
    zero_one_images_train = imageToVector(zero_one_images_train)
    zero_one_images_test, zero_one_labels_test = filterTwoDigits(digit1, digit2, x_test, y_test)
    zero_one_images_test = imageToVector(zero_one_images_test)

    for i in range(len(zero_one_images_train)):
        zero_one_images_train[i] = (zero_one_images_train[i]) / 255

    for i in range(len(zero_one_images_test)):
        zero_one_images_test[i] = (zero_one_images_test[i]) / 255
    zero_one_images_train = np.transpose(np.asarray(zero_one_images_train))
    zero_one_labels_train = np.asarray(zero_one_labels_train)
    zero_one_images_test = np.transpose(np.asarray(zero_one_images_test))
    zero_one_labels_test = np.asarray(zero_one_labels_test)

    w1 = np.zeros(784)
    bet = 0.5
    c = 1e-4
    wStar, w_k = gradient_descent(zero_one_images_train, zero_one_labels_train, w1, 100, 0.25, bet, c, 0.001)
    error_train = calculateError(w_k[len(w_k) - 1], w_k)
    wStar_test, w_k_test = gradient_descent(zero_one_images_test, zero_one_labels_test, w1, 100, 0.25, bet, c, 0.001)
    # error_test = calculateError(min(w_k_test), w_k_test)
    error_test = calculateError(w_k_test[len(w_k) - 1], w_k_test)

    plt.semilogy(error_train, label='train')
    plt.semilogy(error_test, label='test')
    plt.suptitle("Gradient Descent: " + str(digit1) + "/" + str(digit2))
    plt.xlabel("iteration")
    plt.ylabel(r'$|f(w^{(k)}-f(w^*)|$')
    plt.legend()
    plt.show()

def runAndPlotNewton(digit1, digit2):
    zero_one_images_train, zero_one_labels_train = filterTwoDigits(digit1, digit2, x_train, y_train)
    zero_one_images_train = imageToVector(zero_one_images_train)
    zero_one_images_test, zero_one_labels_test = filterTwoDigits(digit1, digit2, x_test, y_test)
    zero_one_images_test = imageToVector(zero_one_images_test)

    # for i in range(len(zero_one_images_train)):
    #     zero_one_images_train[i] = (zero_one_images_train[i]) / 255
    #
    # for i in range(len(zero_one_images_test)):
    #     zero_one_images_test[i] = (zero_one_images_test[i]) / 255
    zero_one_images_train = np.transpose(np.asarray(zero_one_images_train))
    zero_one_labels_train = np.asarray(zero_one_labels_train)
    zero_one_images_test = np.transpose(np.asarray(zero_one_images_test))
    zero_one_labels_test = np.asarray(zero_one_labels_test)

    w1 = np.zeros(784)
    bet = 0.5
    c = 1e-4
    wStar, w_k = exactNewton(zero_one_images_train, zero_one_labels_train, w1, 100, 1, bet, c, 0.0001)
    error_train = calculateError(w_k[len(w_k) - 1], w_k)
    wStar_test, w_k_test = exactNewton(zero_one_images_test, zero_one_labels_test, w1, 100, 1, bet, c, 0.0001)
    # error_test = calculateError(min(w_k_test), w_k_test)
    error_test = calculateError(w_k_test[len(w_k) - 1], w_k_test)

    plt.semilogy(error_train, label='train')
    plt.semilogy(error_test, label='test')
    plt.suptitle("Exact Newton: " + str(digit1) + "/" + str(digit2))
    plt.xlabel("iteration")
    plt.ylabel(r'$|f(w^{(k)}-f(w^*)|$')
    plt.legend()
    plt.show()

runAndPlotGD(0,1)
# runAndPlotGD(8,9)

# runAndPlotNewton(0,1)
runAndPlotNewton(8,9)