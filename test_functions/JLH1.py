import torch
import numpy as np

# 
#    JLH1: 2D objective, 1 constraints
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

def JLH1(individuals): 

    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."

    
    fx = []
    gx = []
    
    for x in individuals:
        test_function = (- (x[0]-0.5)**2 - (x[1]-0.5)**2 ) 
        fx.append(test_function) 
        gx.append( x[0] + x[1] - 0.75 )
        
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))
    gx = torch.tensor(gx)
    gx = torch.reshape(gx, (len(gx),1))
    
    return gx, fx





def JLH1_Scaling(X): 

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    return X









