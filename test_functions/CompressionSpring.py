import torch
import numpy as np


#
#
#   CompressionSpring: 8D objective, 6 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed
#     variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-
#     24):2325–2336
#
#



def CompressionSpring(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."
    
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    n = individuals.size(0)
    

    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        d = x[0]
        D = x[1]
        N = x[2]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( (N+2)*D*d**2 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 1 -  ( D*D*D * N / (71785* d*d*d*d) )
        g2 = (4*D*D - D*d) / (12566 * (D*d*d*d - d*d*d*d)) + 1/(5108*d*d) -  1
        g3 = 1 - 140.45*d / (D*D * N)
        g4 = (D+d)/1.5 - 1

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



def CompressionSpring_Scaling(X): 

    assert torch.is_tensor(X) and X.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."
    
    d = (X[:,0] * ( 1   - 0.05 ) + 0.05 ).reshape(X.shape[0],1)
    D = (X[:,1] * ( 1.3 - 0.25 ) + 0.25   ).reshape(X.shape[0],1)
    N = (X[:,2]  * ( 15  - 2    ) + 2         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((d, D, N), dim=1)

    return X_scaled








