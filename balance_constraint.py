import numpy as np
import torch
from data import DATA_DIR

def balance_constr(Pg, Qg, V, theta, Pd, Qd , DATA_balance, Ybus, device):
    '''
    Pg: Predicted active power generation at each generator
    Qg: Predicted reactive power generation at each generator
    V: Predicted voltage magnitude at each bus
    theta: Predicted voltage angle at each bus
    Ybus: Admittance matrix
    '''   
    n_batch = Pg.shape[0]
    n_bus = V.shape[1]
    # n_gen = Pg.shape[1]
    # n_load = Pd.shape[1]
    
    mask_pg = torch.zeros((n_bus, n_batch), device=device)
    mask_pg[DATA_balance['idx_Gbus'], :] = Pg.transpose(0,1)
    
    mask_qg = torch.zeros((n_bus, n_batch), device=device)
    mask_qg[DATA_balance['idx_Gbus'], :] = Qg.transpose(0,1)
    
    mask_pd = torch.zeros((n_bus, n_batch), device=device)
    mask_pd[DATA_balance['idx_Ldbus'], :] = Pd.transpose(0,1)
    
    mask_qd = torch.zeros((n_bus, n_batch), device=device)
    mask_qd[DATA_balance['idx_Lqbus'], :] = Qd.transpose(0,1)
    
    # for every sample in the batch we create a 3D tensor of size n_bus x n_bus x n_batch which every row is the vector of the voltage magnitude
    # mask1_V = [V1, .., V1; V2, .., V2; ...; Vn, .., Vn]
    #mask2_V = [V1, .., Vn; V1, .., Vn; ...; V1, .., Vn]
    mask1_V = V.transpose(0, 1).repeat(n_bus, 1, 1).to(device)
    mask2_V = V.transpose(0, 1).repeat(n_bus, 1, 1).transpose(0,1).to(device)
    # V_i * V_j = mask2_V * mask1_V
    
    mask1_theta = theta.transpose(0, 1).repeat(n_bus, 1, 1).to(device)
    mask2_theta = theta.transpose(0, 1).repeat(n_bus, 1, 1).transpose(0,1).to(device)

    # theta_i - theta_j = mask2_theta - mask1_theta
    #calculate the power flow equations
    #Pij = V_i * V_j * (Gij * cos(theta_i - theta_j) + Bij * sin(theta_i - theta_j))
    #Qij = V_i * V_j * (Gij * sin(theta_i - theta_j) - Bij * cos(theta_i - theta_j))
    
    Pij = (mask2_V * mask1_V * (Ybus.reshape(n_bus, n_bus, -1).real * torch.cos(mask2_theta - mask1_theta) + 
                                Ybus.reshape(n_bus, n_bus, -1).imag * torch.sin(mask2_theta - mask1_theta)))
    
    Qij = (mask2_V * mask1_V * (Ybus.reshape(n_bus, n_bus, -1).real * torch.sin(mask2_theta - mask1_theta) -
                                Ybus.reshape(n_bus, n_bus, -1).imag * torch.cos(mask2_theta - mask1_theta)))
    
    sigma_Pij = torch.sum(Pij, dim=1) #dimensions: n_bus x n_batch
    sigma_Qij = torch.sum(Qij, dim=1)
    
    balance_P = torch.abs(mask_pg - mask_pd - sigma_Pij).sum(dim=0).reshape(n_batch, 1) #dimentions: n_batch x 1
    balance_Q = torch.abs(mask_qg - mask_qd - sigma_Qij).sum(dim=0).reshape(n_batch, 1)
    
    #constructing the term Vi^2  * Gij in the line flow
    #dimention: n_lines x n_batch
    added_term_p = (Ybus.reshape(n_bus, n_bus, -1).real[DATA_balance['f_buses'], DATA_balance['t_buses'], :]*(mask2_V * mask1_V)[DATA_balance['f_buses'],DATA_balance['f_buses'],:])
    added_term_q = -1 * (Ybus.reshape(n_bus, n_bus, -1).imag[DATA_balance['f_buses'], DATA_balance['t_buses'], :]*(mask2_V * mask1_V)[DATA_balance['f_buses'],DATA_balance['f_buses'],:])
    
    line_flow_p = -1 * Pij[DATA_balance['f_buses'], DATA_balance['t_buses'], :] + added_term_p
    line_flow_q = -1 * Qij[DATA_balance['f_buses'], DATA_balance['t_buses'], :] + added_term_q
    
    line_flow = torch.sqrt(line_flow_p**2 + line_flow_q**2)
    flow_limit = torch.tensor(DATA_balance['fl'], device=device).repeat(n_batch, 1).transpose(0,1) * 0.01

    zero_balance = torch.zeros((n_batch, 1), device=device)
    zero_flow = torch.zeros((DATA_balance['f_buses'].shape[0], n_batch), device=device)
    #print(torch.mean(torch.maximum(balance_P, zero_balance)) )
    #print(torch.mean(torch.maximum(balance_Q, zero_balance)) )
    #print(torch.mean(torch.maximum(line_flow - flow_limit, zero_flow)))
    return (torch.mean(torch.maximum(balance_P, zero_balance)) + torch.mean(torch.maximum(balance_Q, zero_balance)) + torch.mean(torch.maximum(line_flow - flow_limit, zero_flow)))