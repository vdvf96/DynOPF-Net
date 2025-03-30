from datetime import date, datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from scipy.integrate import odeint
from torch.autograd import grad
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import wandb
from functools import reduce
import operator
import itertools
#PYTHONPATH="/Users/vincenzodivitofrancesco/documents/github/PI-ROPF"
from data import DATA_DIR
from utils import read_data
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

#from signal import signal, SIGPIPE, SIG_DFL  
#signal(SIGPIPE,SIG_DFL) 
#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)
#import os
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'


class PINN(nn.Module):
    '''
    :param layers: list of the number of neurons in each layer. 
    #[num of neurons in the input layer, number of neurons in the first hidden layer, ..., number of neurons in the output layer]
    :param activation: activation function
    '''
    def __init__(self, layers: list, activation)->None:
        super().__init__()
        self.layers = layers
        print("Activation: ", activation)
        if activation == "GELU":
            self.activation = nn.GELU
        elif activation == "SELU":
            self.activation = nn.SELU
        elif activation == "CELU":
            self.activation = nn.CELU
        elif activation == "RELU":
            self.activation = nn.ReLU
        elif activation == "ELU":
            self.activation = nn.ELU
        elif activation == "TANH":
            self.activation = nn.Tanh
        elif activation == "SIGMOID":
            self.activation = nn.Sigmoid
        #self.activation = activation
        self.net = self._build_net()

    #This is the function that builds the neural network based on the layers and activation function that the user specifies
    def _build_net(self):
        layers = []
        layer_sizes = [self.layers[0]]
        layer_sizes += (len(self.layers)-2) * [self.layers[1]]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.BatchNorm1d(b), self.activation()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [ nn.Linear(layer_sizes[-1], self.layers[-1]) ]
        for layer in layers :
             if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        return self.net(x).detach().numpy()

    # This function computes the gradients of the neural network with respect to the input
    # The first column of the input x is the time, and the rest of the columns are related to the initial conditions and control inputs
    def gradient(self, x, y_pred):
        dy_dx = grad(outputs=y_pred, inputs=x, grad_outputs=torch.ones_like(y_pred),
                     create_graph = True, allow_unused=True)[0]
        d2y_dx2 = grad(outputs=dy_dx, inputs=x, grad_outputs=torch.ones_like(dy_dx), 
                       create_graph = True, allow_unused=True)[0]
        
        return dy_dx, d2y_dx2


    def loss(self, x, x_init, y_act_init, Ybus, M_i, D_i, coefficient, param): 
        ''' 
        computes the loss function for the PINN
        :param x: input data
        '''
        x.requires_grad = True
        x_init.requires_grad = True

        # x = torch.cat((t, delta_i, omega_i, Pm, Pd, V, theta), dim=1)

        y_pred = self.net(x)
        Pm_i = x[:, 3:4]
        Pd = x[:, 4:5].float()
        V = x[:, 5:5+n_neighbor_bus]
        theta = x[:, 5+n_neighbor_bus:].clone()
        theta[:, 5+n_neighbor_bus+i: 5+n_neighbor_bus+i+1] = y_pred[:, 0:1]
        
        # mask1_V : we have to repeat the V_i for all the neighbors of the bus i
        mask1_V = V[:, i:i+1].repeat(1, n_neighbor_bus)
        # V_i * V_j = mask1_V * V
        
        mask1_theta = theta[:, i:i+1].repeat(1, n_neighbor_bus)
        
        # Pij = V_i * V_j * (Gij * cos(theta_i - theta_j) + Bij * sin(theta_i - theta_j))
        # The form below is the polar form
        
        Pij = (mask1_V * V * (torch.abs(Ybus.repeat(len(x), 1)) * torch.cos(mask1_theta - theta - torch.angle(Ybus.repeat(len(x), 1)))))
        
        ## CHANGE 1: approximation

        # Pij = (mask1_V * V * (torch.abs(Ybus.repeat(len(x), 1)) * (1-.5*(mask1_theta - theta - torch.angle(Ybus.repeat(len(x), 1)))**2)))
        
        # Pij = (V * mask1_V * (Ybus.repeat(len(x), 1).real * torch.cos(mask1_theta - theta) + 
        #                             Ybus.repeat(len(x), 1).imag * torch.sin(mask1_theta - theta)))
        
        P_ij_gPINN = (V * mask1_V * (- Ybus.repeat(len(x), 1).real * torch.sin(mask1_theta - theta) + Ybus.repeat(len(x), 1).imag * torch.cos(mask1_theta - theta)))
        
        # P_ij_gPINN = (V * mask1_V * (- Ybus.repeat(len(x), 1).real * (mask1_theta - theta +  .166*(mask1_theta - theta)**3) + Ybus.repeat(len(x), 1).imag * ( 1 - .5*(mask1_theta - theta)**2)))
        
        Pg_i = torch.sum(Pij, dim=1).reshape(len(x), 1) + Pd     #dimentions: n_batch x n_bus
        Pg_i_gPINN = torch.sum(P_ij_gPINN, dim=1)     #dimentions: n_batch x n_bus
        
        # dy_dt, d2y_dt2 = self.gradient(x, y_pred[:, 0:1])
        dy_dt, d2delta = self.gradient(x, y_pred[:, 0:1])
        d2y_dt2, d2omega = self.gradient(x, y_pred[:, 1:2])
        dy_dt = dy_dt[:, 0:1]
        d2y_dt2 = d2y_dt2[:, 0:1]
        d2delta = d2delta[:, 0:1]
        d2omega = d2omega[:, 0:1]
        
        # M_i * d2delta +  D_i * dy_dt + Pg_i*torch.sin(y_pred[:, 0:1]) - Pm_i
        ODE1 = dy_dt - y_pred[:, 1:2]   ### d(alpha)/dt = w(t)
        # ODE2 =  M_i * d2y_dt2 +  D_i * y_pred[:, 1:2] + Pg_i*torch.sin(y_pred[:, 0:1]) - Pm_i
        # Pg_i = Pg_i[0,:].T.unsqueeze(1)
        
        #ODE2 =  M_i * d2y_dt2 +  D_i * y_pred[:, 1:2] + Pg_i -Pm_i   ### dynamic of the generator, Eq (6)

        ODE2 = 1/(2 * param['H'][1])*(Pm_i - param['D'][1] * 2 * np.pi * d2y_dt2 - V[i] * 1/param['X_d_prime'][1] * torch.sin(y_pred[:, 0:1] - theta[i]))
        
        #print(torch.mean(Pg_i -Pm_i))

        #print(Pg_i[0,:])
        #print(Pg_i[1,:])
        #print(Pg_i[:,0]) ### alpha * np.ones ()

        #print("Inertial term: ",  F.mse_loss( M_i * d2y_dt2, torch.zeros_like(ODE2)))
        #print("First order term: ", F.mse_loss( D_i * y_pred[:, 1:2], torch.zeros_like(ODE2)))
        #print("Powers: ", F.mse_loss( Pg_i - Pm_i, torch.zeros_like(ODE2)))

        dODE1 = d2delta - d2y_dt2
        # dODE2 = M_i * d2omega + D_i * d2y_dt2 + Pg_i*dy_dt*torch.cos(y_pred[:, 0:1])
        dODE2 = M_i * d2omega + D_i * d2y_dt2 + Pg_i_gPINN*dy_dt

        dODE2 = 1/(2 * param['H'][1])*(param['D'][1] * 2 * np.pi * d2y_dt2 - V[i] * 1/param['X_d_prime'][1] * torch.cos(y_pred[:, 1:2] - theta[i])*dy_dt)

        loss_pde1 = F.mse_loss(ODE1, torch.zeros_like(ODE1))
        loss_pde2 = F.mse_loss(ODE2, torch.zeros_like(ODE2))
        loss_dPDE1 = F.mse_loss(dODE1, torch.zeros_like(dODE1))
        loss_dPDE2 = F.mse_loss(dODE2, torch.zeros_like(dODE2))
        
        loss_gov = loss_pde1 + loss_pde2 + loss_dPDE1 + loss_dPDE2
        # loss_gov = coefficient[1] * loss_pde1 + coefficient[2] * loss_pde2 + coefficient[3] * loss_dPDE1 + coefficient[4] * loss_dPDE2
        y_pred_init = self.net(x_init)
        loss_init = F.mse_loss(y_pred_init - y_act_init, torch.zeros_like(y_pred_init))
        # print(f'This is for initial: {loss_init}')
        # print(f'loss gverning equation: {loss_gov}')
        loss = loss_gov + loss_init
        # loss = loss_gov + coefficient[0] * loss_init
        # loss = 0.01*((coefficients[1]/coefficients[0]) * loss_pde1 + (coefficients[1]/coefficients[1])*loss_pde2 + (coefficients[1]/coefficients[2]) * loss_dPDE1 + (coefficients[1]/coefficients[3]) * loss_dPDE2 + (coefficients[1]/coefficients[4]) * loss_init)
        return loss, loss_init.item(), loss_pde1.item(), loss_pde2.item(), loss_dPDE1.item(), loss_dPDE2.item(), loss_pde1, loss_pde2, loss_dPDE1, loss_dPDE2, loss_init, loss_gov

    def train(self):

        param = {'w_s': 2*np.pi*60,

        # 'M':         [23.64/(np.pi*60), 6.4/(np.pi*60), 3.01/(np.pi*60)2
         'M':         [47.28, 12.8, 6.2],
        # 'D':         [0., 0., 0.],
        # 'D':         [0.15, 0.15, 0.1
         'D':         [2, 2, 2],   #From thesis
         #'X_d_prime': [2*0.270, 3.5*0.209, 3.75*0.304],
         'X_d_prime': [0.270, 0.209, 0.304],
         'X_q_prime': [0.470, 0.850, 0.5795],
         'H':         [2.6309, 5.078, 1.200],
        }
        #config = wandb.config
        #x_train, x_test, x_init_train, x_init_test,  Ybus, M_i, D_i ,epochs, lr ,batch_size = config['x_train'], config['x_test'], config['x_init_train'], config['x_init_test'], config['y_actual_train'], config['y_actual_test'], config['Ybus'], float(config['M_i']), float(config['D_i']), config['epochs'], config['learning_rate'], config['batch_size']
        y_act_init_train, y_act_init_test = y_actual_train, y_actual_test
        epochs, lr ,batch_size =  config['epochs'], config['learning_rate'], config['batch_size']
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1)
        train_dataset = TensorDataset(x_train, x_train)
        data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)
        val_dataset = TensorDataset(x_test, x_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # coefficients = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32, device=device)
        # coefficient = torch.tensor([1], dtype=torch.float32, device=device)
        # Introducing weight coefficients for each loss term

        coeff_init = torch.tensor(1.0, dtype=torch.float32, device=device)
        coeff_pde1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        coeff_pde2 = torch.tensor(1.0, dtype=torch.float32, device=device)
        coeff_dpde1 = torch.tensor(1.0, dtype=torch.float32, device=device)
        coeff_dpde2 = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        coeff_terms = []
        epoch_list = []
        best_val_loss = 1000
        max_patience = 150
        patience = 0
        mu = 0.9
        n = 5 ### number of batches between each coeff update
        i = 0
        train_iterator = itertools.cycle(data_loader)
        val_iterator = itertools.cycle(val_loader)

        max_abs_grad = None
        mean_abs_grad_init = None

        batch_losses_val = []
        loss_initial_list_val = []
        loss_pde1_batch_val = []
        loss_pde2_batch_val = []
        loss_dpde1_batch_val = []
        loss_dpde2_batch_val = []

        batch_losses_train = []
        loss_initial_list_train = []
        loss_pde1_batch_train = []
        loss_pde2_batch_train = []
        loss_dpde1_batch_train = []
        loss_dpde2_batch_train = []

        for epoch in range(epochs):
            batch_losses = []
            coefficient = torch.tensor([coeff_init, coeff_pde1, coeff_pde2, coeff_dpde1, coeff_dpde2], device=device)
            #for i, (x, _) in enumerate(data_loader):
            x, _ = next(iter(train_iterator))
            optimizer.zero_grad()
            l, loss_initial, pde1, pde2, dpde1, dpde2, l_pde1, l_pde2, l_dpde1, l_dpde2, l_init, l_gov = self.loss(x, x_init_train, y_act_init_train, Ybus, M_i, D_i, coefficient, param)
            batch_losses_train.append(l.item())
            loss_initial_list_train.append(loss_initial)
            loss_pde1_batch_train.append(pde1)
            loss_pde2_batch_train.append(pde2)
            loss_dpde1_batch_train.append(dpde1)
            loss_dpde2_batch_train.append(dpde2)
            if i%n!=0:
                grads = torch.autograd.grad(l, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] # retain_graph = True)[0]
                grads_init = torch.autograd.grad(l-l_gov, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] #, retain_graph = True)[0]
                #grads_pde1 = torch.autograd.grad(l_pde1, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] #, allow_unused=True, retain_graph = True)[0]
                #grads_dpde1 = torch.autograd.grad(l_dpde1, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] #, allow_unused=True, retain_graph = True)[0]
                #grads_pde2 = torch.autograd.grad(l_pde2, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] #, allow_unused=True, retain_graph = True)[0]
                #grads_dpde2 = torch.autograd.grad(l_dpde2, self.net.parameters(), create_graph=False, allow_unused=True, retain_graph = True) [0] #, allow_unused=True, retain_graph = True)[0]
                # batch_losses.append(loss.item())
                # loss_initial_list.append(loss_initial)
                # loss_pde1_batch.append(pde1)
                # loss_pde2_batch.append(pde2)
                # loss_dpde1_batch.append(dpde1)
                # loss_dpde2_batch.append(dpde2)
                # Compute the maximum of the absolute values of the gradients
                if max_abs_grad==None:
                    max_abs_grad = torch.max(torch.abs(grads))#.float()
                    mean_abs_grad_init = torch.mean(torch.abs(grads_init))#.float()
                else:
                    max_abs_grad += torch.max(torch.abs(grads))#.float()
                    mean_abs_grad_init += torch.mean(torch.abs(grads_init))#.float()
                #mean_abs_grad_pde1 = torch.mean(torch.abs(grads_pde1))
                #mean_abs_grad_dpde1 = torch.mean(torch.abs(grads_dpde1))
                #mean_abs_grad_pde2 = torch.mean(torch.abs(grads_pde2))
                #mean_abs_grad_dpde2 = torch.mean(torch.abs(grads_dpde2))
                #elif max_abs_grad!=None:
                coeff_init = (1-mu*coeff_init) + mu * max_abs_grad/mean_abs_grad_init
            #coeff_pde1 = (1-mu*coeff_pde1) + mu * max_abs_grad/mean_abs_grad_pde1
            #coeff_pde2 = (1-mu*coeff_pde2) + mu * max_abs_grad/mean_abs_grad_pde2
            #coeff_dpde1 = (1-mu*coeff_dpde1) + mu * max_abs_grad/mean_abs_grad_dpde1
            #coeff_dpde2 = (1-mu*coeff_dpde2) + mu * max_abs_grad/mean_abs_grad_dpde2

            loss = coeff_init * l_init + coeff_pde1 * l_pde1 + coeff_pde2 * l_pde2 + coeff_dpde1 * l_dpde1 + coeff_dpde2 * l_dpde2
    
            loss.backward() #retain_graph=True)

            optimizer.step()
            #optimizer.zero_grad()

            #if i%5==0:
            #wandb.log({"Loss/train": loss.item()})
            #wandb.log({"Loss/train_initial": loss_initial})
            #wandb.log({"Loss/train_angle_eq": pde1})
            #wandb.log({"Loss/train_gov_eq": pde2})
            #wandb.log({"Loss/train_d_angle_eq": dpde1})
            #wandb.log({"Loss/train_d_gov_eq": dpde2})
                
            # if i%10==0:
            #     print("Loss/train: ",loss.item()) 
            #print("Loss/train_initial: ", loss_initial)  
            #print("Loss/train_angle_eq", pde1)
            #print("Loss/train_gov_eq" ,pde2)
            #print("Loss/train_d_angle_eq", dpde1)
            #print("Loss/train_d_gov_eq", dpde2)
            
            # training_loss.append(np.mean(batch_losses))
            # loss_init.append(np.mean(loss_initial_list))
            # loss_pde1.append(np.mean(loss_pde1_batch))
            # loss_pde2.append(np.mean(loss_pde2_batch))
            # loss_dpde1.append(np.mean(loss_dpde1_batch))
            # loss_dpde2.append(np.mean(loss_dpde2_batch))
             
            #wandb.log({"Loss/train_per_epoch": np.mean(batch_losses)})
                        
            # scheduler.step()
            # coefficients = torch.mean(torch.tensor(loss_items, dtype=torch.float32, device=device), dim=0)
            
            #if epoch % 1 == 0:
            """The coefficients are updated based on the average of the corresponding loss term's values during each epoch. 
            If the current loss term value is higher than the average, the coefficient is increased, and if it's lower, the coefficient is decreased.
            """
            #coeff_init = coeff_init + pho * np.mean(loss_initial_list)
            #coeff_pde1 *= (1 + 0.05*torch.sign(pde1 - torch.mean(torch.tensor(loss_pde1_batch)))) #*pde1 - torch.mean(torch.tensor(loss_pde1_batch)))
            #coeff_pde2 *= (1 + 0.05*torch.sign(pde2 - torch.mean(torch.tensor(loss_pde2_batch)))) #*pde1 - torch.mean(torch.tensor(loss_pde1_batch)))
            #coeff_dpde1 *= (1 + 0.05*torch.sign(dpde1 - torch.mean(torch.tensor(loss_dpde1_batch)))) #*pde1 - torch.mean(torch.tensor(loss_pde1_batch)))
            #coeff_dpde2 *= (1 + 0.05*torch.sign(dpde2 - torch.mean(torch.tensor(loss_dpde2_batch)))) #*pde1 - torch.mean(torch.tensor(loss_pde1_batch)))

            #wandb.log({"Coeff/coeff_init": coeff_init.item()})
            #wandb.log({"Coeff/coeff_angle_eq": coeff_pde1.item()})
            #wandb.log({"Coeff/coeff_gov_eq": coeff_pde2.item()})
            #wandb.log({"Coeff/coeff_d_angle_eq": coeff_dpde1.item()})
            #wandb.log({"Coeff/coeff_d_gov_eq": coeff_dpde2.item()})

            if coeff_init > 1e4:
                coeff_init = torch.tensor([1.0], dtype=torch.float32, device=device)
            coeff_terms.append(coeff_init.item())
            self.net.eval()

            #for x_val, _ in val_loader:
            x_val, _ = next(iter(val_iterator))
            x_val = x_val.to(device)
            loss_val, loss_initial, pde1, pde2, dpde1, dpde2, l_pde1, l_pde2, l_dpde1, l_dpde2, l_init, l_gov  = self.loss(x_val, x_init_test, y_act_init_test, Ybus, M_i, D_i, coefficient, param)
            batch_losses_val.append(loss_val.item())
            loss_initial_list_val.append(loss_initial)
            loss_pde1_batch_val.append(pde1)
            loss_pde2_batch_val.append(pde2)
            loss_dpde1_batch_val.append(dpde1)
            loss_dpde2_batch_val.append(dpde2)

            #wandb.log({"Loss/val": np.mean(batch_losses)})
            #wandb.log({"Loss/val_initial": np.mean(loss_initial_list)})
            #wandb.log({"Loss/val_angle_eq": np.mean(loss_pde1_batch)})
            #wandb.log({"Loss/val_gov_eq": np.mean(loss_pde2_batch)})
            #wandb.log({"Loss/val_d_angle_eq": np.mean(loss_dpde1_batch)})
            #wandb.log({"Loss/val_d_gov_eq": np.mean(loss_dpde2_batch)})
            if i%10==0:
                print("Val. loss : ", loss_val.item())
            
            #print("Loss/val_initial: ", np.mean(loss_initial_list))
            #print("Loss/val_angle_eq: ", np.mean(loss_pde1_batch))
            #print("Loss/val_gov_eq: ", np.mean(loss_pde2_batch))
            #print("Loss/val_d_angle_eq", np.mean(loss_dpde1_batch))
            #print("Loss/val_d_gov_eq", np.mean(loss_dpde2_batch))

            if  loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item() #np.mean(batch_losses_val)
                print("Best val. loss: ", best_val_loss)
                patience = 0
                torch.save(model.state_dict(), "PINN_"+str(torch.mean(delta_i_init).numpy())+"_"+str(torch.mean(omega_i_init).numpy())+"_.pt")
            else:
                patience += 1
                scheduler.step()
            if patience == max_patience:
                print("Early stopping.")
                print("Number of iteration: ", i)
                break
            i += 1
            #wandb.log({"Loss/val_per_epoch": np.mean(batch_losses_val)})
            epoch_list.append(epoch)
                #logger.info(f'Epoch: {epoch}, Loss_train: {training_loss[-1]:.4f}, loss_val: {val_loss[-1]:.4f}')
            
            # if epoch % 1000 == 0:
            #     np.save(f'loss_train_{current_time}.npy', training_loss)
            #     np.save(f'loss_val_{current_time}.npy', val_loss)
            #     np.save(f'coefficint_{current_time}.npy', coeff_terms)
            #     torch.save(self.state_dict(), f'model-gen_b{i}_{current_time}.pth')   
                
        #logger.info('Model saved')
        
        plt.figure()
        plt.plot(epoch_list, batch_losses_train[:len(epoch_list)], '-k', label='Train', color='red')
        plt.plot(epoch_list, batch_losses_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value')
        plt.legend()
        plt.yscale('log')
        #plt.savefig(f'loss_{current_time}.pdf')
        

        plt.figure()
        #plt.plot(loss_initial_list, '-k', label='Train')
        plt.plot(epoch_list, loss_initial_list_train[:len(epoch_list)], '-k', label='Train', color='red')
        plt.plot(epoch_list, loss_initial_list_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value for initial condition')
        plt.yscale('log')
        plt.legend()
#       plt.savefig(f'loss_initial_{current_time}.pdf')
        
        plt.figure()
        #plt.plot(loss_pde1_batch, '-k', label='Train')
        plt.plot(epoch_list, loss_pde1_batch_train[:len(epoch_list)], '-k', label='Test', color='red')
        plt.plot(epoch_list, loss_pde1_batch_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value for first term')
        plt.yscale('log')
        plt.legend()
#       plt.savefig(f'loss_PDE1_{current_time}.pdf')
        
        plt.figure()
        #plt.plot(loss_pde2_batch, '-k', label='Train')
        plt.plot(epoch_list, loss_pde2_batch_train[:len(epoch_list)], '-k', label='Train', color='red')
        plt.plot(epoch_list, loss_pde2_batch_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value for second term')
        plt.yscale('log')
        plt.legend()
#       plt.savefig(f'loss_PDE2_{current_time}.pdf')
        
        plt.figure()
        #plt.plot(loss_dpde1_batch, '-k', label='Train')
        plt.plot(epoch_list, loss_dpde1_batch_train[:len(epoch_list)], '-k', label='Train', color='red')
        plt.plot(epoch_list, loss_dpde1_batch_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value for derative of first term')
        plt.yscale('log')
        plt.legend()
#       plt.savefig(f'loss_dPDE1_{current_time}.pdf')
        
        plt.figure()
        #plt.plot(loss_dpde2_batch, '-k', label='Train')
        plt.plot(epoch_list, loss_dpde2_batch_train[:len(epoch_list)], '-k', label='Train', color='red')
        plt.plot(epoch_list, loss_dpde2_batch_val[:len(epoch_list)], '--k', label='Test')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss value for derative of second term')
        plt.yscale('log')
        plt.legend()
#       plt.savefig(f'loss_dPDE2_{current_time}.pdf')
        plt.show()
        
#       plt.figure()
#       plt.plot(epoch_list, coeff_terms, '-k', label='loss coefficient')
#       plt.xlabel('Number of Epochs')
#       plt.ylabel('Lagrangian multiplier')
#       plt.legend()
#       plt.savefig(f'coefficint_{current_time}.pdf')
        
global x_train, x_test, x_init_train, x_init_test, y_actual_train, y_actual_test, Ybus, M_i, D_i         
                                    


if __name__ == '__main__':
    # Create a summary writer
    # writer = SummaryWriter('src/PINN_logs')
     
    '''
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='PINN')
    '''

    #wandb.login()
    # Copy your config 
    #config = wandb.config
    torch.manual_seed(123)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    logger.info(f"Today's date: {today} {current_time}")
    
    # parameters of the network
    n_bus = 14
    n_gen = 5
    n_loads = 11
    #perc_inint_cond = 
    # parameters of the generators
    #     M_i = [0.2, 0.2]
    M_i = [12.8, 6.2] #[0.4*2.5, 0.4] #increseing the inertia of the first generator to make it more stable
    #     D_i = [2, 2]
    D_i = [2, 2] #[0.15*8, 0.15] #increseing the damping of the first generator to make it more stable
    # parametrs of the network
    # batch_size = 256
    # n_epoch = 10000
    # lr = 0.001  #for SGD=0.1, for Adam=0.001

    pho = 1 #e-3
    PORTHON_OF_DATA = 0.1
    collocation_point = 0.9999  
    # read the data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, DATA_bounds, DATA_balance, Ybus = read_data(n_bus, n_loads, n_gen)
    print(np.where(DATA_bounds['PG_max']!=0)[0])

    for i in (np.where(DATA_bounds['PG_max']!=0)[0])[:1]:
        input_dim = 1 + 2 + 1 + 1 + 2 * (len(torch.where(Ybus[i, :] != 0)[0]))  # time (1) + initial conditions (2) + Pm (1) + Pd (1) + 2 * neighboring_bus (V, theta)
        n_neighbor_bus = len(torch.where(Ybus[i, :] != 0)[0])
        # layers = [input_dim, 150, 100, 50, 25, 2]
        # layers = [input_dim, 20, 20, 20, 20, 2]  
        # logger.info(f'Number of epochs: {config.epochs}, batch size: {config.batch_size}, portion of the data: {config.PORTHON_DATA_Init}, number of collacation point:{config.Collocation}, learning rate: {config.learning_rate}')
        #logger.info(f'Generator {i+1}') 
        t = torch.linspace(0, 10, 99990).reshape(-1, 1)
        # print(t.size())
        delta_i = 1*(torch.pi*torch.rand((len(t), 1)) - 0.5*torch.pi) #previously 0.25
        # print(delta_i.size())
        omega_i = 1*(torch.pi*torch.rand((len(t), 1)) - 0.5*torch.pi)
        #Pm = torch.tensor(data[f'gen{i.item()+1}:pg'].shift(-1).values[:int(collocation_point*len(data))], dtype=torch.float32).reshape(-1, 1) #for having some samples to be unstable 
        Pm = 1*torch.normal(torch.tensor(data[f'gen{i.item()+1}:pg'].values[:int(collocation_point*len(data))], dtype=torch.float32).reshape(-1, 1), std = 0.1) ### CHANGE 1 - Pm_i is sampled from N(pg_i, 0.1)
        
        Pd_fake = torch.tensor(data.iloc[:int(collocation_point*len(data)), :n_loads].values, dtype=torch.float32)
        Pd = torch.zeros((len(Pd_fake), n_bus))
        Pd[:, DATA_balance['idx_Ldbus']] = Pd_fake
        Pd = Pd[:, i].reshape(-1, 1)

        V = torch.tensor(data.iloc[:int(collocation_point*len(data)), 2*n_loads+2*n_gen:(2*n_loads+2*n_gen+n_bus)].values, dtype=torch.float32)[:, torch.where(Ybus[i, :] != 0)[0]]
        theta = torch.tensor(data.iloc[:int(collocation_point*len(data)), (2*n_loads+2*n_gen+n_bus):].values, dtype=torch.float32)[:, torch.where(Ybus[i, :] != 0)[0]]
        con_input = torch.cat((Pm, Pd, V, theta), dim=1)
        con_input = con_input.repeat(int(t.shape[0]/con_input.shape[0]), 1)
        x = torch.cat((t, delta_i, omega_i, con_input), dim=1)
        t_init = torch.zeros((int(PORTHON_OF_DATA*len(data)), 1))

        beta = .1 #.000 #scale the range of delta's initial condition interval
        delta_i_init = beta*(torch.pi*0.5*torch.rand((len(t_init), 1))) #.sort(dim=0)[0] + 1.5*torch.pi) ### CHANGE 2 - delta_i_init is derived from delta_i , delta_i.sort(dim=0)[0][:len(t_init)] #
        omega_i_init = beta*(torch.pi*0.5*torch.rand((len(t_init), 1))) #.sort(dim=0)[0] + 1.5*torch.pi) ### omega_i.sort(dim=0)[0][:len(t_init)] #2*torch.pi*torch.rand((len(t_init), 1)).sort(dim=0)[0] - torch.pi ### CHANGE 3 - omega_i_init is derived from omega_i 
        
        #print(torch.mean(delta_i_init))
        #print(torch.mean(omega_i_init))

        Pm_init = 1*torch.normal(torch.tensor(data[f'gen{i.item()+1}:pg'].values[:int(PORTHON_OF_DATA*len(data))], dtype=torch.float32).reshape(-1, 1), std = 0.1) ### CHANGE 1 - Pm_i is sampled from N(pg_i, 0.1)
        # Pm_init = torch.tensor(data[f'gen{i.item()+1}:pg'].shift(-1).values[:int(PORTHON_OF_DATA*len(data))], dtype=torch.float32).reshape(-1, 1) #for having some samples to be unstable
        Pd_init_fake = torch.tensor(data.iloc[:int(PORTHON_OF_DATA*len(data)), :n_loads].values, dtype=torch.float32)
        Pd_init = torch.zeros((len(Pd_init_fake), n_bus))
        Pd_init[:, DATA_balance['idx_Ldbus']] = Pd_init_fake
        Pd_init = Pd_init[:, i].reshape(-1, 1)
        V_init = torch.tensor(data.iloc[:int(PORTHON_OF_DATA*len(data)), 2*n_loads+2*n_gen:(2*n_loads+2*n_gen+n_bus)].values, dtype=torch.float32)[:, torch.where(Ybus[i, :] != 0)[0]]
        theta_init = torch.tensor(data.iloc[:int(PORTHON_OF_DATA*len(data)), (2*n_loads+2*n_gen+n_bus):].values, dtype=torch.float32)[:, torch.where(Ybus[i, :] != 0)[0]]
        # init_cond = torch.cat((delta_i_init, omega_i_init), dim=1)
        con_input_init = torch.cat((Pm_init, Pd_init, V_init, theta_init), dim=1)
        x_init = torch.cat((t_init, delta_i_init, omega_i_init, con_input_init), dim=1)
        y_actual = torch.cat((delta_i_init, omega_i_init), dim=1)
        # split the collocation points into training and testing
        x_train, x_test , _, _ = train_test_split(x, x ,test_size=0.2, random_state=1)
        # x_train, x_test = x[:int(0.8 * len(x))], x[int(0.8 * len(x)):]
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        # split the initial points into training and testing
        x_init_train, x_init_test, y_actual_train, y_actual_test = train_test_split(x_init, y_actual, test_size=0.2, random_state=1)
        # x_init_train, x_init_test = x_init[:int(0.8 * len(x_init))], x_init[int(0.8 * len(x_init)):]
        y_actual_train, y_actual_test = y_actual[:int(0.8 * len(y_actual))], y_actual[int(0.8 * len(y_actual)):]   
        x_init_train, x_init_test = x_init_train.to(device), x_init_test.to(device)
        y_actual_train, y_actual_test = y_actual_train.to(device), y_actual_test.to(device)
        con_input = con_input.to(device)
        Ybus = Ybus[i, torch.where(Ybus[i, :] != 0)[0]]

    learning_rate_list = [1e-2] # , 1e-4]
    layer_size_list = [20] #, 20, 50, 100]
    activ_function_list = ["TANH"] # ,"GELU", "SIGMOID"]
    num_hidden_layer_list = [2] #, 3, 5, 10]
    batch_size_list = [64]

    M_i = M_i[i]
    D_i = D_i[i]

    #print(arch_list)
    for learning_rate in learning_rate_list:
        for layer_size in layer_size_list:
            for activ_function in activ_function_list:
                for num_hidden_layer in num_hidden_layer_list:
                    for bs in batch_size_list:
                        arch_list = [layer_size] * num_hidden_layer
                        arch_list.insert(0,11) 
                        arch_list.append(2)
                        config = {
                                    'learning_rate': learning_rate,
                                    'epochs': 500,
                                    'batch_size': bs,
                                    'PORTHON_DATA_Init': 0.1,
                                    'Collocation': 0.9999,
                                    'architecture': arch_list,
                                    'activ_function': activ_function,
                                    'delta_i_init': torch.mean(delta_i_init).numpy(),
                                    'omega_i_init': torch.mean(omega_i_init).numpy()
                            }
                        '''
                        wandb.init(
                            project="PINN",
                            name = str(layer_size) + "_" + str(num_hidden_layer) + "_" + str(learning_rate) + "_" + activ_function + "_" + str(bs),
                            config = {
                                    'learning_rate': learning_rate,
                                    'epochs': 10000,
                                    'batch_size': bs,
                                    'PORTHON_DATA_Init': 0.1,
                                    'Collocation': 0.9999,
                                    'architecture': arch_list,
                                    'activ_function': activ_function
                            }
                        )
                        '''

                        '''
                            'x_train': x_train,
                            'x_test': x_test,
                            'x_init_train': x_init_train,
                            'x_init_test': x_init_test,
                            'y_actual_train': y_actual_train,
                            'y_actual_test': y_actual_test,
                            'Ybus': Ybus,
                            'M_i': M_i[i], 
                            'D_i': D_i[i]
                        '''

                        model = PINN(config['architecture'], config['activ_function'])
                        #print(model.net)

                        #wandb.watch(model, log="all", log_graph=True)
                        #wandb.watch(model.net, log="all")
                        model = model.to(device)
                        model.train()
                        #wandb.agent(sweep_id, model.train)

                    #wandb.finish()
                    break
    
    model.load_state_dict(torch.load("PINN_"+str(torch.mean(delta_i_init).numpy())+"_"+str(torch.mean(omega_i_init).numpy())+"_.pt"))
    idx = torch.randint(0, len(Pm), (1,)).item()
    delta = data.iloc[idx, 2*n_loads+2*n_gen+n_bus + i]
    omega = 0
    Pm_ = Pm[idx]
    Pd_ = Pd[idx]
    V_ = V[idx, :]
    theta_ = theta[idx, :]     
    con_input_ = torch.cat((Pm_.reshape(1,-1), Pd_.reshape(1,-1), V_.reshape(1,-1) , theta_.reshape(1,-1)), dim=1)
    init_cond_ = torch.tensor([delta, omega], dtype=torch.float32).reshape(1, -1)
    
    param = {'w_s': 2*np.pi*60,

        # 'M':         [23.64/(np.pi*60), 6.4/(np.pi*60), 3.01/(np.pi*60)2
         'M':         [47.28, 12.8, 6.2],
        # 'D':         [0., 0., 0.],
        # 'D':         [0.15, 0.15, 0.1
         'D':         [2, 2, 2],   #From thesis
         #'X_d_prime': [2*0.270, 3.5*0.209, 3.75*0.304],
         'X_d_prime': [0.270, 0.209, 0.304],
         'X_q_prime': [0.470, 0.850, 0.5795],
         'H':         [2.6309, 5.078, 1.200],
        }
    
    def model_ode(U,t):
    # Here U is a vector such that delta=U[0] and w=U[1]. This function should return [delta', w']
        Pg = 0
        for i in range(n_neighbor_bus):
            if i == 0:
                Pg += V_[0].item() * V_[0].item()*(Ybus.real[i].item())
            else:
                Pg += V_[0].item() * V_[i].item()*(Ybus.real[i].item() * np.cos(U[0] - theta_[i].item())+ Ybus.imag[i].item() * np.sin(U[0] - theta_[i].item()))
        Pg = Pg + Pd_
        #return torch.tensor([U[1], (-D_i / M_i) * U[1] - (1 / M_i) * Pg + (Pm_ / M_i)])
        return torch.tensor([param['w_s']*(U[1]), 1/(2 * param['H'][1])*(Pm_ - param['D'][1] * 2 * np.pi * U[1] - V_[i] * 1/param['X_d_prime'][1] * np.sin(U[0] - theta_[i]))])  
    
    # delta_,  omega_, theta_, V_

    # init_cond_ = torch.tensor([delta, omega], dtype=torch.float32).reshape(1, -1)
    # con_input_ = torch.cat((Pm_.reshape(1,-1), Pd_.reshape(1,-1), V_.reshape(1,-1) , theta_.reshape(1,-1)), dim=1)
    # con_input_init = torch.cat((Pm_init, Pd_init, V_init, theta_init), dim=1)
    # x_init = torch.cat((t_init, delta_i_init, omega_i_init, con_input_init), dim=1)

    final_t = 10
    t = np.linspace(0, final_t, 1000)
    Uinit = torch.tensor([.1*torch.rand(1), .05*torch.rand(1)]).cpu()
    print("Initial condition: ", Uinit)
    Us = odeint(model_ode, Uinit, t)
    delta_ = Us[:,0]
    omega_ = Us[:,1]

    # for i in range(20):
    #     print("Delta = ",delta_[i*1000])
    #     print("Omega = ",omega_[i*1000])

    intermediate_t = 10 #0.5
    t_val = torch.linspace(0, intermediate_t, 1000).reshape((-1, 1))
    delta_t = t[1] - t[0]

    con_input_ = con_input_.repeat(int(t_val.shape[0]/con_input_.shape[0]), 1)
    init_cond_ = init_cond_.repeat(int(t_val.shape[0]/init_cond_.shape[0]), 1)
    x_val = torch.cat((t_val, init_cond_, con_input_), dim=1)
    x_val = x_val.to(device)
    
    y_pred_delta = []
    y_pred_omega = []
    a = 0.5

    for _ in range(int(final_t/intermediate_t)):
        with torch.no_grad():
            y_val = model(x_val)

        t_val = torch.linspace(a, a+intermediate_t, 1000).reshape((-1, 1))
        x_val = torch.cat((t_val, y_val[-1, :].repeat(t_val.shape[0], 1), con_input_), dim=1).to(device)
        #predicted_delta_dot = torch.zeros_like(y_val)
        #predicted_omega_dot = torch.zeros_like(y_val)

        #predicted_delta_dot[0,0] = (y_val[1,0] - y_val[0,0])/ (1 * delta_t)
        #predicted_omega_dot[0,0] = (y_val[1,1] - y_val[0,1])/ (1 * delta_t)

        #predicted_delta_dot[1:,0] = (y_val[1:,0] - y_val[:-1,0])/ (1 * delta_t)
        #predicted_omega_dot[1:,0] = (y_val[1:,1] - y_val[:-1,1])/ (1 * delta_t)
        
        y_pred_delta.extend(y_val[:,0].cpu().numpy().tolist())
        y_pred_omega.extend(y_val[:,1].cpu().numpy().tolist())
        # y_pred_delta.extend(predicted_delta_dot.cpu().numpy().reshape((-1,)).tolist())
        # y_pred_omega.extend(predicted_omega_dot.cpu().numpy().reshape((-1,)).tolist())
        a += intermediate_t

    y_pred_delta = [a for a in y_pred_delta]
    y_pred_omega = [a for a in y_pred_omega]

    np.save('delta_pred_PINN.npy', np.array(y_pred_delta))
    np.save('omega_pred_PINN.npy', np.array(y_pred_omega))

    
    plt.figure()
    plt.plot(t, y_pred_delta, label='$\delta$')
    plt.plot(t, y_pred_omega, label='$\omega$')
    plt.plot(t, delta_, label = '$\delta_{act}$')
    plt.plot(t, omega_, label = '$\omega_{act}$')
    plt.xlabel('Time')
    plt.ylabel('$\delta$')
    plt.legend()
    #plt.savefig(f'developed PINN_{current_time}.pdf')
    plt.show()