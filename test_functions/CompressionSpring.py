import torch
import numpy as np

#
#
#   CompressionSpring: 8D objective, 6 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-24):2325–2336
#
#


def CompressionSpring(individuals):

    assert (
        torch.is_tensor(individuals) and individuals.size(1) == 3
    ), "Input must be an n-by-3 PyTorch tensor."

    fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i, :]

        d, D, N = x

        ## Negative sign to make it a maximization problem
        test_function = -((N + 2) * D * d**2)
        fx = torch.cat((fx, torch.tensor([[test_function]])))

        ## Calculate and add constraint terms
        gx1 = torch.cat((gx1, torch.tensor([[1 - (D * D * D * N / (71785 * d * d * d * d))]])))
        gx2 = torch.cat((gx2, torch.tensor([[(4 * D * D - D * d) / (12566 * (D * d * d * d - d * d * d * d)) + 1 / (5108 * d * d) - 1]])))
        gx3 = torch.cat((gx3, torch.tensor([[1 - 140.45 * d / (D * D * N)]])))
        gx4 = torch.cat((gx4, torch.tensor([[(D + d) / 1.5 - 1]])))

    gx = torch.cat((gx1, gx2, gx3, gx4), 1)

    return gx, fx


def CompressionSpring_Scaling(X):

    assert (
        torch.is_tensor(X) and X.size(1) == 3
    ), "Input must be an n-by-3 PyTorch tensor."

    scale_mult = torch.tensor([(1 - 0.05), (1.3 - 0.25), (15 - 2)])
    scale_add = torch.tensor([0.05, 0.25, 2])

    return torch.mul(X, scale_mult) + scale_add
