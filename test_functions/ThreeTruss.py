import torch
import numpy as np


#
#
#   ThreeTruss: 2D objective, 3 constraints
#
#   Reference:
#     Yang XS, Hossein Gandomi A (2012) Bat algo-
#      rithm: a novel approach for global engineer-
#      ing optimization. Engineering computations
#      29(5):464–483
#
#


def ThreeTruss(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    

    fx = []
    gx1 = []
    gx2 = []
    gx3 = []


    n = individuals.size(0)
    
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        
        if x1 <=1e-5:
            x1 = 1e-5
        if x2 <=1e-5:
            x2 = 1e-5
        
        L = 100
        P = 2
        sigma = 2
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 2*np.sqrt(2)*x1 + x2 ) * L
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = ( np.sqrt(2)*x1 + x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma
        g2 = ( x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma
        g3 = ( 1 ) / (x1 + np.sqrt(2)*x2) * P - sigma
    
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
       
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3), 1)


    return gx, fx






def ThreeTruss_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."

    return X
























































