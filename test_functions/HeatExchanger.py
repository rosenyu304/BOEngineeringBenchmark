import torch
import numpy as np

#
#
#   HeatExchanger: 8D objective, 6 constraints
#
#   Reference:
#     Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global
#     engineering optimization. Engineering computations 29(5):464–483
#
#


def HeatExchanger(individuals):

    assert (
        torch.is_tensor(individuals) and individuals.size(1) == 8
    ), "Input must be an n-by-8 PyTorch tensor."

    fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
        torch.tensor([]),
    ) * 12

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i, :]

        x1, x2, x3, x4, x5, x6, x7, x8 = x

        ## Negative sign to make it a maximization problem
        test_function = -(x1 + x2 + x3)

        fx = torch.cat((fx, torch.tensor([[test_function]])))

        ## Calculate constraints terms
        gx1 = torch.cat((gx1, torch.tensor([[0.0025 * (x4 + x6) - 1]])))
        gx2 = torch.cat((gx2, torch.tensor([[0.0025 * (x5 + x7 - x4) - 1]])))
        gx3 = torch.cat((gx3, torch.tensor([[0.01 * (x8 - x5) - 1]])))
        gx4 = torch.cat((gx4, torch.tensor([[833.33252 * x4 + 100 * x1 - x1 * x6 - 83333.333]])))
        gx5 = torch.cat((gx5, torch.tensor([[1250 * x5 + x2 * x4 - x2 * x7 - 125 * x4]])))
        gx6 = torch.cat((gx6, torch.tensor([[x3 * x5 - 2500 * x5 - x3 * x8 + 125 * 10000]])))

    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6), 1)

    return gx, fx


def HeatExchanger_Scaling(X):

    assert (
        torch.is_tensor(X) and X.size(1) == 8
    ), "Input must be an n-by-8 PyTorch tensor."

    scale_mult = torch.tensor(
        [
            (10000 - 100),
            (10000 - 100),
            (10000 - 100),
            (1000 - 10),
            (1000 - 10),
            (1000 - 10),
            (1000 - 10),
            (1000 - 10),
        ]
    )
    scale_add = torch.tensor([100, 1000, 1000, 10, 10, 10, 10, 10])

    return torch.mul(X, scale_mult) + scale_add
