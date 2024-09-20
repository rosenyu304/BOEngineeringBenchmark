import torch
import numpy as np


#
#
#   ThreeTruss: 2D objective, 3 constraints
#
#   Reference:
#     Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global
#     engineering optimization. Engineering computations 29(5):464–483
#
#


def ThreeTruss(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."

    fx, gx1, gx2, gx3 = (torch.tensor([]),) * 4

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i,:]

        x1, x2 = x

        if x1 <=1e-5:
            x1 = 1e-5
        if x2 <=1e-5:
            x2 = 1e-5

        L = 100
        P = 2
        sigma = 2

        ## Negative sign to make it a maximization problem
        test_function = - ( 2*np.sqrt(2)*x1 + x2 ) * L
        fx = torch.cat((fx, torch.tensor([[test_function]])))

        ## Calculate constraints terms
        gx1 = torch.cat((gx1, torch.tensor([[( np.sqrt(2)*x1 + x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma]])))
        gx2 = torch.cat((gx2, torch.tensor([[( x2 ) / (np.sqrt(2)*x1*x1 + 2*x1*x2) * P - sigma]])))
        gx3 = torch.cat((gx3, torch.tensor([[( 1 ) / (x1 + np.sqrt(2)*x2) * P - sigma]])))

    gx = torch.cat((gx1, gx2, gx3), 1)

    return gx, fx


def ThreeTruss_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 2, "Input must be an n-by-2 PyTorch tensor."

    return X
