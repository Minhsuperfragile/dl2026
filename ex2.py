x, y = [], []

with open('price.csv', 'r') as f:
    t = [l.strip().split(",") for l in f.readlines()]
    x , y = [float(t_[0]) for t_ in t] , [float(t_[1]) for t_ in t]

def forward(x:list[float] , w:list[float]): 
    return [x_ * w[0] + w[1] for x_ in x]

def gradient(x:list[float], w:list[float], y:list[float]) : 
    y_pred = forward(x, w)
    
    # l = 1/2N * (y_pred - y)**2 
    # dL/dw_0 = 1/N * (y_pred - y) * x
    grad_0 = 1/len(x) * sum([(y_p - y_) * x_ for y_p, y_, x_ in zip(y_pred, y, x)])

    # dL/dw_1 = 1/N * (y_pred - y) 
    grad_1 = 1/len(x) * sum([y_p - y_ for y_p, y_ in zip(y_pred, y)])

    return grad_0, grad_1

def loss(x:list[float], w:list[float], y:list[float]):
    y_pred = forward(x, w)
    return 1/(2*len(x)) * sum([(y_p - y_)**2 for y_p, y_ in zip(y_pred, y)])

def backward(x:list[float], y:list[float], iter = 10, init:list[float] = [0,0], lr = 5*10**(-4) ): 
    for i in range(iter): 
        l = loss(x, init, y)
        w_0 , w_1 = gradient(x,init,y)
        init[0] -= w_0 * lr
        init[1] -= w_1 * lr 
        # print(f'{i}, {l:2f}, {init}')

    return init

w = backward(x, y, iter=10000)

from matplotlib import pyplot as plt

plt.scatter(x,y, color = 'red') 
plt.plot([10, 80], forward([10,80], w), color= 'green')
plt.title(f"Linear regression: {w[0]:.2f} * x + {w[1]:.2f}")
plt.show()