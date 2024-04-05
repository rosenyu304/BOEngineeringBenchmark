import torch
import numpy as np

#
#
#   GKXWC2: 2D objective, 1 constraints
#
#   Reference:
#       Gardner JR, Kusner MJ, Xu ZE, et al (2014)
#       Bayesian optimization with inequality con-
#       straints. In: ICML, pp 937–945
#
#

def GKXWC2(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    fx = []
    gx = []
    
    for x in individuals:
        
        g = np.sin(x[0])*np.sin(x[1]) + 0.95
        fx.append( - np.sin(x[0]) - x[1]  ) # maximize -(x1^2 +x 2^2)
        gx.append( g )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx


def GKXWC2_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    X_scaled = X*6;
    return X_scaled







































