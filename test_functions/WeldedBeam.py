import torch
import numpy as np

#
#
#   WeldedBeam: 4D objective, 5 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed
#     variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-
#     24):2325–2336
#
#

def WeldedBeam(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."
    
    
    C1 = 1.10471
    C2 = 0.04811
    C3 = 14.0
    fx = torch.zeros(individuals.shape[0], 1)
    gx1 = torch.zeros(individuals.shape[0], 1)
    gx2 = torch.zeros(individuals.shape[0], 1)
    gx3 = torch.zeros(individuals.shape[0], 1)
    gx4 = torch.zeros(individuals.shape[0], 1)
    gx5 = torch.zeros(individuals.shape[0], 1)
    
    for i in range(individuals.shape[0]):
        
        x = individuals[i,:]

        h = x[0]
        l = x[1]
        t = x[2]
        b = x[3]
        
        test_function = - ( C1*h*h*l + C2*t*b*(C3+l) )
        fx[i] = test_function
        
        ## Calculate constraints terms 
        tao_dx = 6000 / (np.sqrt(2)*h*l)
        
        tao_dxx = 6000*(14+0.5*l)*np.sqrt( 0.25*(l**2 + (h+t)**2 ) ) / (2* (0.707*h*l * ( l**2 /12 + 0.25*(h+t)**2 ) ) )
        
        tao = np.sqrt( tao_dx**2 + tao_dxx**2 + l*tao_dx*tao_dxx / np.sqrt(0.25*(l**2 + (h+t)**2)) )
        
        sigma = 504000/ (t**2 * b)
        
        P_c = 64746*(1-0.0282346*t)* t * b**3
        
        delta = 2.1952/ (t**3 *b)
        
        
        ## Calculate 5 constraints
        g1 = (-1) * (13600- tao) 
        g2 = (-1) * (30000 - sigma) 
        g3 = (-1) * (b - h)
        g4 = (-1) * (P_c - 6000) 
        g5 = (-1) * (0.25 - delta)
        
        gx1[i] =  g1        
        gx2[i] =  g2     
        gx3[i] =  g3             
        gx4[i] =  g4 
        gx5[i] =  g5 
    
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5), 1)
    return gx, fx


def WeldedBeam_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."
    
    h = (X[:,0]  * (10-0.125) + 0.125 ).reshape(X.shape[0],1)
    l = (X[:,1]  * (15-0.1  ) + 0.1   ).reshape(X.shape[0],1)
    t = (X[:,2]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    b = (X[:,3]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((h, l, t, b), dim=1)
    return X_scaled








