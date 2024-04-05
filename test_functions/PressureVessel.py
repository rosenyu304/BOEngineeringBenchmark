import torch
import numpy as np

#
#
#   PressureVessel: 4D objective, 4 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed
#     variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-
#     24):2325–2336
#
#


def PressureVessel(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."
    
    C1 = 0.6224
    C2 = 1.7781
    C3 = 3.1661
    C4 = 19.84
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    n = individuals.size(0)
    

    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        Ts = x[0]
        Th = x[1]
        R = x[2]
        L = x[3]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( C1*Ts*R*L + C2*Th*R*R + C3*Ts*Ts*L + C4*Ts*Ts*R )
        fx.append(test_function) 
        

        g1 = -Ts + 0.0193*R
        g2 = -Th + 0.00954*R
        g3 = (-1)*np.pi*R*R*L + (-1)*4/3*np.pi*R*R*R + 750*1728
        g4 = L-240
        
        

        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx = torch.cat((gx1, gx2, gx3, gx4), 1)

    
    return gx, fx






def PressureVessel_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."
    
    Ts  = (X[:,0] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    Th  = (X[:,1] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    R   = (X[:,2] * (200-10) + 10).reshape(X.shape[0],1)
    L   = (X[:,3] * (200-10) ).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((Ts, Th, R, L), dim=1)
    
    return X_scaled





































































