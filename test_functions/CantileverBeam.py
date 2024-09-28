import torch
import numpy as np


r"""

    CantileverBeam: 10D objective, 11 constraints

    Reference:
      Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for
      global engineering optimization. Engineering computations 29(5):464–483


"""


def CantileverBeam(individuals):

    assert (
        torch.is_tensor(individuals) and individuals.size(1) == 10
    ), "Input must be an n-by-10 PyTorch tensor."

    fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
        torch.tensor([]),
    ) * 12

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i, :]

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x

        P = 50000
        E = 2 * 107
        L = 100

        ## Negative sign to make it a maximization problem
        test_function = -(
            x1 * x6 * L + x2 * x7 * L + x3 * x8 * L + x4 * x9 * L + x5 * x10 * L
        )
        fx = torch.cat((fx, torch.tensor([[test_function]])))

        ## Calculate constraints terms
        gx1 = torch.cat((gx1, torch.tensor([[600 * P / (x5 * x10 * x10) - 14000]])))
        gx2 = torch.cat((gx2, torch.tensor([[6 * P * (L * 2) / (x4 * x9 * x9) - 14000]])))
        gx3 = torch.cat((gx3, torch.tensor([[6 * P * (L * 3) / (x3 * x8 * x8) - 14000]])))
        gx4 = torch.cat((gx4, torch.tensor([[6 * P * (L * 4) / (x2 * x7 * x7) - 14000]])))
        gx5 = torch.cat((gx5, torch.tensor([[6 * P * (L * 5) / (x1 * x6 * x6) - 14000]])))
        gx6 = torch.cat((gx6, torch.tensor([[P * L**3 * (1 / L + 7 / L + 19 / L + 37 / L + 61 / L) / (3 * E) - 2.7]])))
        gx7 = torch.cat((gx7, torch.tensor([[x10 / x5 - 20]])))
        gx8 = torch.cat((gx8, torch.tensor([[x9 / x4 - 20]])))
        gx9 = torch.cat((gx9, torch.tensor([[x8 / x3 - 20]])))
        gx10 = torch.cat((gx10, torch.tensor([[x7 / x2 - 20]])))
        gx11 = torch.cat((gx11, torch.tensor([[x6 / x1 - 20]])))

    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11), 1)

    return gx, fx


def CantileverBeam_Scaling(X):

    assert (
        torch.is_tensor(X) and X.size(1) == 10
    ), "Input must be an n-by-10 PyTorch tensor."

    scale_mult = torch.tensor(
        [
            (5 - 1),
            (5 - 1),
            (5 - 1),
            (5 - 1),
            (5 - 1),
            (65 - 30),
            (65 - 30),
            (65 - 30),
            (65 - 30),
            (65 - 30),
        ]
    )
    scale_add = torch.tensor([1, 1, 1, 1, 1, 30, 30, 30, 30, 30])

    return torch.mul(X, scale_mult) + scale_add
