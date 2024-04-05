import torch
import numpy as np

#
#
#   HeatExchanger: 8D objective, 6 constraints
#
#   Reference:
#     Yang XS, Hossein Gandomi A (2012) Bat algo-
#      rithm: a novel approach for global engineer-
#      ing optimization. Engineering computations
#      29(5):464–483
#
#



def HeatExchanger(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 8, "Input must be an n-by-8 PyTorch tensor."
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    gx5 = []
    gx6 = []
    gx7 = []
    gx8 = []
    gx9 = []
    gx10 = []
    gx11 = []
    

    n = individuals.size(0)
    

    for i in range(n):
        
        x = individuals[i,:]
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( x1+x2+x3 )

        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 0.0025 * (x4+x6) - 1
        g2 = 0.0025 * (x5 + x7 - x4) - 1
        g3 = 0.01 *(x8-x5) - 1
        g4 = 833.33252*x4 + 100*x1 - x1*x6 - 83333.333
        g5 = 1250*x5 + x2*x4 - x2*x7 - 125*x4
        g6 = x3*x5 - 2500*x5 - x3*x8 + 125*10000

        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
    
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
    
    gx5 = torch.tensor(gx5)  
    gx5 = gx1.reshape((n, 1))

    gx6 = torch.tensor(gx6)  
    gx6 = gx2.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6), 1)


    return gx, fx





def HeatExchanger_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 8, "Input must be an n-by-8 PyTorch tensor."
    
    x1 = (X[:,0] * (10000-100) + 100).reshape(X.shape[0],1)
    x2 = (X[:,1] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x3 = (X[:,2] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x4 = (X[:,3] * (1000-10) + 10).reshape(X.shape[0],1)
    x5 = (X[:,4] * (1000-10) + 10).reshape(X.shape[0],1)
    x6 = (X[:,5] * (1000-10) + 10).reshape(X.shape[0],1)
    x7 = (X[:,6] * (1000-10) + 10).reshape(X.shape[0],1)
    x8 = (X[:,7] * (1000-10) + 10).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
    
    return X_scaled







































































