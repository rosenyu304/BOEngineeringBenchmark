import torch
import numpy as np

#
#
#   KeaneBump: N-D objective (can take data of different dimention; we use 18), 
#              2 constraints
#
#   Reference:
#        Keane A (1994) Experiences with optimizers in
#        structural design. In: Proceedings of the con-
#        ference on adaptive computing in engineering
#        design and control, pp 14–27
#
#



def KeaneBump(X): 

    
    
    fx = torch.zeros(X.shape[0], 1).to(torch.float64)
    gx1 = torch.zeros(X.shape[0], 1).to(torch.float64)
    gx2 = torch.zeros(X.shape[0], 1).to(torch.float64)
    
    
    
    for i in range(X.shape[0]):
        x = X[i,:]
        
        cos4 = 0
        cos2 = 1
        sq_denom = 0
        
        pi_sum = 1
        sigma_sum = 0
        
        for j in range(X.shape[1]):
            cos4 += torch.cos(x[j]) ** 4
            cos2 *= torch.cos(x[j]) ** 2
            sq_denom += (j+1) * (x[j])**2
            
            pi_sum *= x[j]
            sigma_sum += x[j]
        
        
        # Objective
        test_function = torch.abs(  (cos4 - 2*cos2) / torch.sqrt(sq_denom)  )
        fx[i] = test_function

        # Constraints
        gx1[i] = 0.75 - pi_sum
        gx2[i] = sigma_sum - 7.5* (X.shape[1])
        
    gx = torch.cat((gx1, gx2), 1)
    return gx, fx





def KeaneBump_Scaling(X):
    
    X_scaled = X*10
    
    return X_scaled





















