import torch
import numpy as np

#
#
#   ReinforcedConcreteBeam: 3D objective, 9 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed
#     variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-
#     24):2325–2336
#
#


def ReinforcedConcreteBeam(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    n = individuals.size(0)
    

    for i in range(n):
        
        x = individuals[i,:]
        
        As = x[0]
        h = x[1]
        b = x[2]
        
        
        test_function = - ( 29.4*As + 0.6*b*h )
        fx.append(test_function) 
        
        g1 = h/b - 4
        g2 = 180 + 7.35*As*As/b - As*h
        
        gx1.append( g1 )       
        gx2.append( g2 )    

       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2), 1)

    
    
    return gx, fx
    
    
    
    
def ReinforcedConcreteBeam_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."
    
    As = (X[:,0] * (15-0.2) + 0.2).reshape(X.shape[0],1)
    b  = (X[:,1] * (40-28)  +28).reshape(X.shape[0],1)
    h  = (X[:,2] * 5 + 5).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((As, b, h), dim=1)
    return X_scaled    
    
    
    
    
    
    
    
    
    
    
    
    
