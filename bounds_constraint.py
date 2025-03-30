import numpy as np
import torch

from data import DATA_DIR


def bound_constraint(Pg, Qg, V, theta, Data, device):
    '''
    x: Input data
    y: Output data
    init_cond: Initial conditions
    num_init_cond: Number of initial conditions
    '''
    n_batch = Pg.shape[0]
    n_bus = V.shape[1]
    n_gen = Pg.shape[1]

    # define the upper and lower bounds for the power generation 
    Pg_max = torch.tensor(Data['PG_max'], device=device).repeat(n_batch, 1)*0.01 #due to the p.u. conversion
    Pg_min = torch.tensor(Data['PG_min'], device=device).repeat(n_batch, 1)*0.01
    
    #defne the upper and lower bounds for the reactive power generation
    Qg_max = torch.tensor(Data['Q_max'], device=device).repeat(n_batch, 1)*0.01
    Qg_min = torch.tensor(Data['Q_min'], device=device).repeat(n_batch, 1)*0.01
    
    #define the upper and lower bounds for the voltage magnitude
    V_min = torch.tensor(Data['V_min'], device=device).repeat(n_batch, 1)
    V_max = torch.tensor(Data['V_max'], device=device).repeat(n_batch, 1)
    
    #define the upper and lower bounds for the angle
    theta_min = torch.zeros(n_batch, n_bus, device=device).fill_(-np.pi/3)
    theta_max = torch.zeros(n_batch, n_bus, device=device).fill_(np.pi/3)
    
    zero_p = torch.zeros(n_batch, n_gen, device=device)
    zero_v = torch.zeros(n_batch, n_bus, device=device)

    return torch.mean(torch.maximum(Pg - Pg_max, zero_p)) + torch.mean(torch.maximum(Pg_min - Pg, zero_p)) + \
        + torch.mean(torch.maximum(Qg - Qg_max, zero_p)) + torch.mean(torch.maximum(Qg_min - Qg, zero_p)) + \
            + torch.mean(torch.maximum(V - V_max, zero_v)) + torch.mean(torch.maximum(V_min - V, zero_v)) +\
                torch.mean(torch.maximum(theta-theta_max, zero_v)) + torch.mean(torch.maximum(theta_min-theta, zero_v))    #+  add the theta constraint


def get_Pg_bounded(Pg, Data, device):
    n_batch = Pg.shape[0]
    # define the upper and lower bounds for the power generation 
    Pg_max = torch.tensor(Data['PG_max'], device=device).repeat(n_batch, 1)*0.01 #due to the p.u. conversion
    Pg_min = torch.tensor(Data['PG_min'], device=device).repeat(n_batch, 1)*0.01
    return (Pg_max - Pg_min) * Pg*2 + Pg_min*2

def get_Qg_bounded(Qg, Data, device):
    n_batch = Qg.shape[0]
    # define the upper and lower bounds for the power generation 
    Qg_max = torch.tensor(Data['Q_max'], device=device).repeat(n_batch, 1)*0.01 #due to the p.u. conversion
    Qg_min = torch.tensor(Data['Q_min'], device=device).repeat(n_batch, 1)*0.01
    return (Qg_max - Qg_min) * Qg *1.1 + Qg_min

def get_Vg_bounded(V, Data, device):
    n_batch = V.shape[0]
    # define the upper and lower bounds for the power generation 
    V_min = torch.tensor(Data['V_min'], device=device).repeat(n_batch, 1)
    V_max = torch.tensor(Data['V_max'], device=device).repeat(n_batch, 1)
    return (V_max - V_min) * V + V_min*1.1



    