### Logistic regression from scratch
import math

x, y, z= [], [], []

with open('loan2.csv', 'r') as f:
    t = [l.strip().split(",") for l in f.readlines()][1:]
    x , y, z = [float(t_[0]) for t_ in t] , [float(t_[1]) for t_ in t], [float(t_[2]) for t_ in t]

def sigmoid(x): 
    return 1/(1+math.exp(-x))

def forward(x, y, w): 
    return [ (x_ * w[0] + y_ * w[1] + w[2]) for x_, y_ in zip(x,y)]

def loss(x, y, w, z) : 
    y_pred = forward(x,y,w)
    L = sum([- (z_ * math.log(sigmoid(y_p)) + (1-z_) * math.log(1-sigmoid(y_p))) for y_p, z_ in zip(y_pred, z)])
    return L / len(x)

def gradient(x, y, w, z): 
    y_pred = forward(x, y , w)
    grad_0 = sum([-z_ * x_ + x_ * (1 - sigmoid(-y_p) ) for x_, z_ , y_p in zip(x,z,y_pred)]) / len(x)

    grad_1 = sum([-z_ * y_ + y_ * (1 - sigmoid(-y_p)) for y_, z_, y_p in zip(y,z,y_pred)]) / len(x)

    grad_2 = sum([1 - z_ - sigmoid(-y_p) for z_, y_p in zip(z, y_pred)]) / len(x)

    return grad_0, grad_1, grad_2 

def backward(x,y,z, init=[0,1,2],lr=10**(-1),iter=10**(3) ):
    for i in range(iter): 
        L = loss(x,y,init,z)
        g_0, g_1, g_2 = gradient(x,y,init,z)
        init[0] -= g_0 * lr
        init[1] -= g_1 * lr
        init[2] -= g_2 * lr

        print(i, L , init)

    return init

w = backward(x,y,z)
    
from matplotlib import pyplot as plt

# plot the line
x1 = [i for i in range(0 , 4) ]
x2 = [ - (x_ * w[0] + w[2]) / w[1] for x_ in x1]

plt.scatter(x,y, c = z)
plt.plot(x1,x2)
plt.ylim(min(y), max(y))
plt.show()