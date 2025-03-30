import numpy as np
import pandas as pd
import torch
from data import DATA_DIR

# function that reads the files and returns the data
def read_data(n_bus, n_loads, n_gen):
    
    DATA_bounds = {}
    DATA_balance = {}
    gendata = pd.read_csv(DATA_DIR/ f'mpcgen{n_bus}.csv',header=None)
    branchdata = pd.read_csv(DATA_DIR/ f'mpcbranch{n_bus}.csv',header=None)
    busdata = pd.read_csv(DATA_DIR/ f'mpcbus{n_bus}.csv',header=None)

    # Generation limits
    DATA_bounds['PG_min'] = np.array(gendata[:][9])
    DATA_bounds['PG_max'] = np.array(gendata[:][8])

    # Generation Reactive limits
    DATA_bounds['Q_min'] = np.array(gendata[:][4])
    DATA_bounds['Q_max'] = np.array(gendata[:][3])
    
    # Voltage limits
    DATA_bounds['V_min'] = np.array(busdata[:][12])
    DATA_bounds['V_max'] = np.array(busdata[:][11])
    
    DATA_balance['idx_Gbus'] =  torch.tensor(np.array([int(i-1) for i in gendata[:][0]])) #due to the difference indexing between MATPOWER and Python
    DATA_balance['idx_Ldbus'] = torch.tensor(np.where(np.array(busdata[:][2]) != 0)[0])
    #DATA_balance['idx_Lqbus'] = torch.tensor(np.where(np.array(busdata[:][3]) != 0)[0])
    DATA_balance['idx_Lqbus'] = torch.tensor(np.where(np.array(busdata[:][2]) != 0)[0])
    
    n_lines = len(branchdata[:][0])       #number of lines
    
    # Incidence matrix. This matrix has size nlines x nbus, with each row having two nonzero elements: 
    # +1 in the entry for the "from" bus of the corresponding line and -1 in the entry for the "to" bus.
    # Inc = np.loadtxt(open(DATA_DIR/ f'Inc{n_bus}.csv', "rb"), delimiter=",", skiprows=0)
    # DATA_balance['Inc'] = torch.tensor(Inc, device=device)
    DATA_balance['f_buses'] = torch.tensor(np.array([int(i-1) for i in branchdata[:][0]])) #?
    #DATA_balance['t_buses'] = torch.tensor(np.array([int(i-1) for i in branchdata[:][1]])) #?
    DATA_balance['t_buses'] = torch.tensor(np.array([int(i-1) for i in branchdata[:][1]])) #?
    print(DATA_balance['f_buses'])
    print(DATA_balance['t_buses'])
    #read the Y_bus matrix
    Ybus = pd.read_csv(DATA_DIR /f'Ybus{n_bus}.csv', header=None)
    Ybus.replace('i', 'j', regex=True, inplace=True)
    Ybus = Ybus.astype('complex').to_numpy()
    Ybus = torch.tensor(Ybus)
    
    # flow limits
    fl = np.array(branchdata[:][5])
    for i in range(0,n_lines):
        if(fl[i] == 0): # matpower assigns "no line limits" to appear as 0
            fl[i] = 1000
    DATA_balance['fl'] = fl
    
    #preparing the input and output data
    if n_bus == 9:
        data = pd.read_csv(DATA_DIR / f'dataset.csv', header = None)
        input = data[data.columns[torch.where(torch.tensor(data.iloc[0, :] != 0)  )[0]][:2*n_loads]]
        Voltages = data[data.columns[3*n_bus:(4*n_bus)]]
        output = pd.concat([data[data.columns[4*n_bus:(4*n_bus+2*n_gen)]], Voltages, data[data.columns[2*n_bus:(3*n_bus)]]], axis=1, join='inner')
    else:
        data = pd.read_csv(DATA_DIR / f'pglib_opf_case{n_bus}_ieee.csv')
        input = data[data.columns[0:2*n_loads]]
        Voltages = data[data.columns[(2*n_loads+3*n_gen):(2*n_loads+3*n_gen+n_bus)]]
        for i in range(len(Voltages.columns)):
            if type (Voltages[Voltages.columns[i]][0]) == str:
                Voltages[Voltages.columns[i]] = Voltages[Voltages.columns[i]].str.split(' ').str.join('')
        
        Voltages = Voltages.astype('complex')
        abs_voltage = Voltages.abs()
        angle_voltage = -(180/np.pi) *  Voltages.apply(np.angle)  #due to the problem in the data, we need to multiply by -1
        output = pd.concat([data[data.columns[2*n_loads:(2*n_loads+2*n_gen)]], abs_voltage, angle_voltage], axis=1, join='inner')
    
    data = pd.concat([input, output], axis=1, join='inner')
    return data, DATA_bounds, DATA_balance, Ybus

if __name__ == "__main__":
    # data, DATA_bounds, DATA_balance,  Ybus = read_data(14, 11, 5, 'cpu')
    data, DATA_bounds, DATA_balance,  Ybus = read_data(9, 3, 3)
    print(f'This is the data: {data}')
    print(f'This is the DATA_bounds: {DATA_bounds}')
    print(f'This is the Ybus: {Ybus}')