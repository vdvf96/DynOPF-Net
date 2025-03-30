from datetime import date, datetime
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from loguru import logger
# from scipy.integrate import odeint
from torch.autograd import grad
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import argparse
import csv

class SwingNN(nn.Module):
    '''
    :param layers: list of the number of neurons in each layer. 
    #[num of neurons in the input layer, number of neurons in the first hidden layer, ..., number of neurons in the output layer]
    :param activation: activation function
    '''
    def __init__(self, layers: list, activation=nn.Softplus)->None:
        super().__init__()
        self.layers = layers
        if activation == "SOFTPLUS":
            self.activation = nn.Softplus
        elif activation == "RELU":
            self.activation = nn.ReLU
        elif activation == "TANH":
            self.activation = nn.Tanh
        elif activation == "SIGMOID":
            self.activation = nn.Sigmoid
        self.net = self._build_net()

    #This is the function that builds the neural network based on the layers and activation function that the user specifies
    def _build_net(self):
        layers = []
        for j in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[j], self.layers[j + 1]))
            if j < len(self.layers) - 2:
                layers.append(self.activation())
        return nn.Sequential(*layers)
    
    def forward(self, t, x):
        return self.net(x)
    
    def predict(self, t, x):
        return self.net(x).detach().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PI_ROPF')
    ### NN HYPERPARAMS
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--nHiddenUnit', type=int, default = 50, help='number of hidden units')
    parser.add_argument('--activation', type=str, default = "RELU", help='activation_function')
    parser.add_argument('--optimizer', type=int, default = 2, help='GD algorithm')
    parser.add_argument('--lr', type=float, default = 1e-3, help='total number of datapoints')
    parser.add_argument('--batchsize', type=int, default = 100, help='training batch size')
    parser.add_argument('--initSplit', type=float, default = .1)
    parser.add_argument('--normalize', type=bool, default = False)
    parser.add_argument('--useFFLayer', type=bool, default = True)
    parser.add_argument('--k', type=float, default = .1)
    parser.add_argument('--k_check', type=float, default = 1)
    parser.add_argument('--delta_k', type=float, default = .1)
    parser.add_argument('--delta_k_epoch', type=int, default = 100)
    parser.add_argument('--delta_k_split', type=float, default = 1)
    parser.add_argument('--max_patience', type=int, default = 10)
    parser.add_argument('--nLayer', type=int, default = 5, help='number of layers')
    parser.add_argument('--id', type=int, default = 5, help='number of layers')

    # gen(1,2,3): nHiddenUnit:50, activation:RELU, lr:1e-3, batchsize: 100, k:.1, T = 1[s]

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    
    torch.manual_seed(args['seed'])

    param = {
        'w_s': 2*np.pi*60,
        'M':         [.1, .1, .1],
        'D':         [2.364, 1.28, 0.903],   #From thesis
        'X_d_prime': [0.0608, 0.1198, 0.1813],
        'H':         [23.64, 6.4, 3.01],
    }

    Pm_ = 0.7565
    V_min = 0.95
    V_max = 1.04

    def get_data(y_zero, t):
        
        theta_ = torch.zeros(1)
        V_ = (V_max - V_min) * torch.rand(1) + V_min #torch.rand(1)
        
        def model_ode(U,t):
            #  Here U is a vector such that delta=U[0] and w=U[1]. This function should return [delta', w']
            # We set Pm = 0.7565
            return [param['w_s']*(U[1]), 1/(2 * param['H'][0])*(Pm_ -param['D'][0] * 2 * np.pi * U[1] - V_[0].item() * 1/param['X_d_prime'][0] * np.sin(U[0] - theta_[0].item()))]  
        sol = odeint(model_ode, y_zero, t.squeeze())
        return sol

    n_samples = 100
    batch_size = args['batchsize']
    t = torch.linspace(0, 10, 1000).reshape(-1, 1)

    X_tensor = torch.zeros([n_samples,4])
    Y_tensor = torch.zeros([n_samples,t.shape[0],4])

    '''
    for i in range(n_samples):
        y_zero = torch.tensor([.1*torch.rand(1),.1*torch.rand(1)])
        X_tensor[i,:] = y_zero
        y_true = get_data(y_zero, t) # #, method='rk4')
        Y_tensor[i,:] = torch.from_numpy(y_true)
        plt.figure()
        plt.plot(t.numpy(), y_true[:,0], label = "$\delta(t)$")
        plt.legend()
        plt.figure()
        plt.plot(t.numpy(), y_true[:,1], label = "$\omega(t)$")
        plt.show()
    '''
    n_gen = 3
    X_tensor = torch.load(f"./new_data_lip_57/X_tensor_gen_4th{n_gen}-stable.pt") ## [delta, omega, theta, V]
    Y_tensor = torch.load(f"./new_data_lip_57/Y_tensor_gen_4th{n_gen}-stable.pt") ## [delta, omega, 0, 0]
    
    #print(torch.max(X_tensor[:,0]))
    #print(torch.max(X_tensor[:,1]))
    #print(torch.min(X_tensor[:,0]))
    #print(torch.min(X_tensor[:,1])) 
    print(torch.max(X_tensor[:,2]))
    print(torch.max(X_tensor[:,3]))
    print(torch.min(X_tensor[:,2]))
    print(torch.min(X_tensor[:,3])) 
    # for i in range(60):
    #     if Y_tensor[i,-1,0]<torch.pi/2:
    #         #plt.plot([i for i in range(1000)], Y_tensor[i,:,1], label = "$\omega(t)$")
    #         plt.plot([i/100 for i in range(1000)], Y_tensor[i,:,0], label = "$\delta(t)$")
    #         plt.legend()
    #         plt.xticks(np.arange(0, 10, 1.0))
    #         plt.title(f"Generator {n_gen}")
    #         plt.grid(True)
    #         plt.show()

    stab = 0
    for i in range(10000):
        if Y_tensor[i,-1,0]<torch.pi/2:
            stab += 1
    
    print(stab/10000)
    '''
        plt.figure()
        #plt.plot([i for i in range(1000)], Y_tensor[0,:,3])
        #plt.plot([i for i in range(1000)], Y_tensor[0,:,2])
        plt.plot([i for i in range(1000)], Y_tensor[i,:,1], label = "$\omega(t)$")
        plt.plot([i for i in range(1000)], Y_tensor[i,:,0], label = "$\delta(t)$")
        plt.legend()
        plt.show()
    
    
    plt.figure()
    #plt.plot([i for i in range(1000)], Y_tensor[1,:,3])
    #plt.plot([i for i in range(1000)], Y_tensor[1,:,2])
    plt.plot([i for i in range(1000)], Y_tensor[1,:,1], label = "$\omega(t)$")
    plt.plot([i for i in range(1000)], Y_tensor[1,:,0], label = "$\delta(t)$")
    plt.legend()
    plt.show()
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor , test_size=0.2, random_state=1)

    train_data = TensorDataset(X_train, Y_train)   # X:(1024,2) Y:(1024)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(X_test, Y_test)   # X:(1024,2) Y:(1024)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    arch_list = [args['nHiddenUnit']] * args['nLayer']
    arch_list.insert(0,4) 
    arch_list.append(4)

    func = SwingNN(arch_list, args['activation']) #ODEFunc()
    lr = args['lr']
    optimizer = optim.AdamW(func.parameters(), lr=lr)
    loss_train, loss_valid = [], []

    patience = 0
    max_patience = 100
    best = 100
    T = 1
    k = args['k']
    k_check = args['k_check']
    delta_k = args['delta_k']
    delta_k_epoch = args['delta_k_epoch']
    delta_k_split = args['delta_k_split']
    loss_train, loss_test = [], []
    k_list_train, delta_k_list_train = [], []
    k_list_test, delta_k_list_test = [], []


    for epoch in range(5000):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            '''
            t_init_1 = np.random.randint(1,len(t)/2-k)
            t_end_1 = t_init_1+k
            t_init_2 = np.random.randint(len(t)/2,len(t)-k)
            t_end_2 = t_init_2+k
            t_tmp = t[t_init_1:t_end_1]+t[t_init_2:t_end_2]
            '''
            y_pred = odeint(func, x, t.squeeze()[:int(k*len(t))], method='rk4')
            #print(x[0,:])
            #print(y[0,0,:])
            y_pred = torch.swapaxes(y_pred, 0, 1)
            #y = y[:,:,:2]
            #y_pred = y_pred[:,:,:2]
            loss = torch.mean((y_pred - y[:,:int(k*len(t)),:])**2)
            if i%50==0:
                np.random.seed(epoch)
                tmp = np.random.randint(1,y.shape[0])
                '''
                plt.figure()
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
                plt.legend()
                plt.show()
                #plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),2], label = "$\Theta(t)$")
                #plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),2].detach().numpy(), label = "$\hat \Theta(t)$")
                #plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),3], label = "$V(t)$")
                #plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),3].detach().numpy(), label = "$\hat V(t)$")
                plt.legend()
                plt.show()
                np.random.seed(i+10)
                tmp = np.random.randint(1,y.shape[0])
                plt.figure()
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
                plt.legend()
                np.random.seed(i+100)
                tmp = np.random.randint(1,y.shape[0])
                plt.figure()
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
                plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
                plt.legend()
                plt.show()
                '''
            print(f"Iteration {i} : training loss = {loss.item()}")
            loss_train.append(loss.item())
            k_list_train.append(k)
            delta_k_list_train.append(delta_k)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: training loss = {loss.item()}')

        func.eval()

        for i, (x, y) in enumerate(test_loader):
            y_pred = odeint(func, x, t.squeeze(), method='rk4')
            y_pred = torch.swapaxes(y_pred, 0, 1)
            #y = y[:,:,:2]
            #y_pred = y_pred[:,:,:2]
            loss = torch.mean((y_pred[:,:int(k*len(t)),:] - y[:,:int(k*len(t)),:])**2)
            
            np.random.seed(i)
            tmp = np.random.randint(epoch,y.shape[0])
            '''
            plt.figure()
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
            plt.plot(t.numpy(), y[tmp,:,0], label = "$\delta(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,0].detach().numpy(), label = "$\hat \delta(t)$")
            plt.plot(t.numpy(), y[tmp,:,1], label = "$\omega(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,1].detach().numpy(), label = "$\hat \omega(t)$")
            #plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),2], label = "$\Theta(t)$")
            #plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),2].detach().numpy(), label = "$\hat \Theta(t)$")
            #plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),3], label = "$V(t)$")
            #plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),3].detach().numpy(), label = "$\hat V(t)$") 
            plt.legend()
            plt.show()
            tmp = np.random.randint(epoch+1,y.shape[0])
            plt.figure()
            plt.plot(t.numpy(), y[tmp,:,0], label = "$\delta(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,0].detach().numpy(), label = "$\hat \delta(t)$")
            plt.plot(t.numpy(), y[tmp,:,1], label = "$\omega(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,1].detach().numpy(), label = "$\hat \omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),2], label = "$\Theta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),2].detach().numpy(), label = "$\hat \Theta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),3], label = "$V(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),3].detach().numpy(), label = "$\hat V(t)$")
            plt.legend()
            tmp = np.random.randint(epoch+2,y.shape[0])
            plt.figure()
            plt.plot(t.numpy(), y[tmp,:,0], label = "$\delta(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,0].detach().numpy(), label = "$\hat \delta(t)$")
            plt.plot(t.numpy(), y[tmp,:,1], label = "$\omega(t)$")
            plt.plot(t.numpy(), y_pred[tmp,:,1].detach().numpy(), label = "$\hat \omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),0], label = "$\delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),0].detach().numpy(), label = "$\hat \delta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),1], label = "$\omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),1].detach().numpy(), label = "$\hat \omega(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),2], label = "$\Theta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),2].detach().numpy(), label = "$\hat \Theta(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y[tmp,:int(k*len(t)),3], label = "$V(t)$")
            # plt.plot(t.numpy()[:int(k*len(t))], y_pred[tmp,:int(k*len(t)),3].detach().numpy(), label = "$\hat V(t)$")
            plt.legend()
            plt.show()
            '''
            print(f'Epoch {epoch}: validation loss = {loss.item()}')
        loss_test.append(loss.item())
        k_list_test.append(k)
        delta_k_list_test.append(delta_k)

        if (epoch+1)%delta_k_epoch==0:
            k=T
        if loss.item()<best and k==T:
            best = loss.item()
            best_model = torch.save(func.state_dict(), f"new_best_model_{n_gen}_{T*10}s_long.pt")
        elif k==T:
            patience += 1
            if patience>= max_patience:
                break

    run_id = args["id"]

    with open(f"loss_train_{run_id}.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "k", "delta_k"])
        for epoch, loss in enumerate(loss_train, 1):
            writer.writerow([epoch, loss, k_list_train[epoch-1], delta_k_list_train[epoch-1]])
    with open(f"loss_valid_{run_id}.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Loss", "k", "delta_k"])
        for epoch, loss in enumerate(loss_test, 1):
            writer.writerow([epoch, loss, k_list_test[epoch-1], delta_k_list_test[epoch-1]])