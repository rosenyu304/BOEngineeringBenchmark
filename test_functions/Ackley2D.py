import torch
import numpy as np
from botorch.test_functions import Ackley
device = torch.device("cpu")
dtype = torch.double

#
#
#   Ackley2D: 2D objective, 2 constraints
#
#   Reference:
#        Eriksson D, Poloczek M (2021) Scalable con-
#        strained bayesian optimization. In: Interna-
#       tional Conference on Artificial Intelligence and
#        Statistics, PMLR, pp 730–738
#
#

def Ackley2D(individuals): 
    
    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    #############################################################################
    #############################################################################
    # Set function here:
    dimm = 2
    fun = Ackley(dim=dimm, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    #############################################################################
    #############################################################################
    
    
    n = individuals.size(0)

    fx = fun(individuals)
    fx = fx.reshape((n, 1))

    #############################################################################
    ## Constraints
    gx1 = torch.sum(individuals,1)  # sigma(x) <= 0 
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.norm(individuals, p=2, dim=1)-5  # norm_2(x) -3 <= 0
    gx2 = gx2.reshape((n, 1))

    gx = torch.cat((gx1, gx2), 1)
    #############################################################################
    
    
    return gx, fx




def Ackley2D_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."
    
    X_scaled = X*15-5
    
    return X_scaled





















