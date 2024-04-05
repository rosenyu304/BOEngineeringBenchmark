import torch
import numpy as np

def PenaltyFunction(gx, fx, rho): # This should be the test function
    #############################################################################
    #############################################################################
    ## Adding penalty function
    n = fx.shape[0]
    N_GX = gx.shape[1]
    arr_to_be_mult_rho = torch.zeros((n,N_GX))

    for i in range(N_GX):
        zero_gx = torch.cat((torch.zeros((n,1)), gx[:,i].reshape((n,1))), 1)
        max_rslt = torch.max(zero_gx, 1)
        max_rslt = max_rslt.values
        
        arr_to_be_mult_rho[:,i] = max_rslt**2


    arr_to_be_mult_rho = torch.sum(arr_to_be_mult_rho, dim=1)
    arr_to_be_mult_rho = arr_to_be_mult_rho 
    #############################################################################
    #############################################################################
    
    
    #############################################################################
    fx_new = fx - rho * arr_to_be_mult_rho.reshape((n,1))
    #############################################################################
    
    return fx_new