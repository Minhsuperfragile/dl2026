def forward(x):
    return x*x

def gradient(x):
    return 2*x

def descent(forward, gradient, init, lr=10**(-2)):
    value = [1,0]
    time = 0
    while ((forward(value[1]) - forward(value[0]))**2)**(1/2) >= lr:
        print(time, ": ", init, " ", forward(init)) 
        time +=1
        init -= gradient(init) * lr
        value[0] = value[1]
        value[1] = init

descent(forward, gradient, init = 10)