import torch
import numpy as np


# 
#    JLH2: 2D objective, 1 constraints
# 
# 
#    Reference:
#        Jetton C, Li C, Hoyle C (2023) Constrained
#        bayesian optimization methods using regres-
#        sion and classification gaussian processes as
#        constraints. In: International Design Engi-
#        neering Technical Conferences and Computers
#        and Information in Engineering Conference,
#        American Society of Mechanical Engineers, p
#        V03BT03A033
# 
#


def JLH2(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    fx = []
    gx = []
    
    for x in individuals:
        
        ## Negative sign to make it a maximization problem
        test_function = - ( np.cos(2*x[0])*np.cos(x[1]) +  np.sin(x[0]) ) 
        
        fx.append(test_function) 
        gx.append( ((x[0]+5)**2)/4 + (x[1]**2)/100 -2.5 )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx



def JLH2_Scaling(individuals): 

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."

    
    X = individuals
    X1 = X[:,0].reshape(X.size(0),1)
    X1 = X1*5-5
    X2 = X[:,1].reshape(X.size(0),1)
    X2 = X2*10-5
    X_scaled = torch.tensor(np.concatenate((X1,X2), axis=1))
    
    return X_scaled










