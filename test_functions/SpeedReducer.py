import torch
import numpy as np

#
#
#   SpeedReducer: 7D objective, 9 constraints
#
#   Reference:
#     Yang XS, Hossein Gandomi A (2012) Bat algorithm: a novel approach for global
#     engineering optimization. Engineering computations 29(5):464–483
#
#


def SpeedReducer(individuals):

    assert (
        torch.is_tensor(individuals) and individuals.size(1) == 7
    ), "Input must be an n-by-7 PyTorch tensor."

    fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
        torch.tensor([]),
    ) * 12

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i, :]

        b, m, z, L1, L2, d1, d2 = x

        C1 = 0.7854 * b * m * m
        C2 = 3.3333 * z * z + 14.9334 * z - 43.0934
        C3 = 1.508 * b * (d1 * d1 + d2 * d2)
        C4 = 7.4777 * (d1 * d1 * d1 + d2 * d2 * d2)
        C5 = 0.7854 * (L1 * d1 * d1 + L2 * d2 * d2)

        ## Negative sign to make it a maximization problem
        test_function = -(C1 * (C2) - C3 + C4 + C5)

        fx = torch.cat((fx, torch.tensor([[test_function]])))

        ## Calculate constraints terms
        gx1 = torch.cat((gx1, torch.tensor([[27 / (b * m * m * z) - 1]])))
        gx2 = torch.cat((gx2, torch.tensor([[397.5 / (b * m * m * z * z) - 1]])))
        gx3 = torch.cat((gx3, torch.tensor([[1.93 * L1**3 / (m * z * d1**4) - 1]])))
        gx4 = torch.cat((gx4, torch.tensor([[1.93 * L2**3 / (m * z * d2**4) - 1]])))
        gx5 = torch.cat((gx5, torch.tensor([[np.sqrt((745 * L1 / (m * z)) ** 2 + 1.69 * 1e6)/ (110 * d1**3) - 1]])))
        gx6 = torch.cat((gx6,torch.tensor([[np.sqrt((745 * L2 / (m * z)) ** 2 + 157.5 * 1e6) / (85 * d2**3) - 1]])))
        gx7 = torch.cat((gx7, torch.tensor([[m * z / 40 - 1]])))
        gx8 = torch.cat((gx8, torch.tensor([[5 * m / (b) - 1]])))
        gx9 = torch.cat((gx9, torch.tensor([[b / (12 * m) - 1]])))

    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9), 1)

    return gx, fx


def SpeedReducer_Scaling(X):

    assert (
        torch.is_tensor(X) and X.size(1) == 7
    ), "Input must be an n-by-7 PyTorch tensor."

    scale_mult = torch.tensor(
        [
            (3.6 - 2.6),
            (0.8 - 0.7),
            (28 - 17),
            (8.3 - 7.3),
            (8.3 - 7.3),
            (3.9 - 2.9),
            (5.5 - 5),
        ]
    )
    scale_add = torch.tensor([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])

    return torch.mul(X, scale_mult) + scale_add
