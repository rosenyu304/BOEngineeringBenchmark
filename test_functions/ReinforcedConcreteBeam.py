import torch
import numpy as np

#
#
#   ReinforcedConcreteBeam: 3D objective, 9 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-24):2325–2336
#
#

def ReinforcedConcreteBeam(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."

    fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i,:]

        As, h, b = x

        test_function = - ( 29.4*As + 0.6*b*h )
        fx = torch.cat((fx, torch.tensor([[test_function]])))

        gx1 = torch.cat((gx1, torch.tensor([[h/b - 4]])))
        gx2 = torch.cat((gx2, torch.tensor([[180 + 7.35*As*As/b - As*h]])))


    gx = torch.cat((gx1, gx2), 1)

    return gx, fx


def ReinforcedConcreteBeam_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 3, "Input must be an n-by-3 PyTorch tensor."

    scale_mult = torch.tensor([(15-0.2), (40-28), (5)])
    scale_add = torch.tensor([0.2, 28, 5])

    return torch.mul(X, scale_mult) + scale_add
