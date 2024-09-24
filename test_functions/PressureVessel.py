import torch
import numpy as np

#
#
#   PressureVessel: 4D objective, 4 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-24):2325–2336
#
#


def PressureVessel(individuals):

    assert (
        torch.is_tensor(individuals) and individuals.size(1) == 4
    ), "Input must be an n-by-4 PyTorch tensor."

    C1, C2, C3, C4 = (0.6224, 1.7781, 3.1661, 19.84)
    fx, gx1, gx2, gx3, gx4 = (torch.tensor([]),) * 5

    n = individuals.size(0)

    for i in range(n):

        x = individuals[i, :]

        Ts, Th, R, L = x

        ## Negative sign to make it a maximization problem
        test_function = -(
            C1 * Ts * R * L + C2 * Th * R * R + C3 * Ts * Ts * L + C4 * Ts * Ts * R
        )
        fx = torch.cat((fx, torch.tensor([[test_function]])))

        gx1 = torch.cat((gx1, torch.tensor([[-Ts + 0.0193 * R]])))
        gx2 = torch.cat((gx2, torch.tensor([[-Th + 0.00954 * R]])))
        gx3 = torch.cat((gx3, torch.tensor([[(-1) * np.pi * R * R * L + (-1) * 4 / 3 * np.pi * R * R * R + 750 * 1728]])))
        gx4 = torch.cat((gx4, torch.tensor([[L - 240]])))

    gx = torch.cat((gx1, gx2, gx3, gx4), 1)

    return gx, fx


def PressureVessel_Scaling(X):

    assert (
        torch.is_tensor(X) and X.size(1) == 4
    ), "Input must be an n-by-4 PyTorch tensor."

    scale_mult = torch.tensor([(98 * 0.0625), (98 * 0.0625), (200 - 10), (200 - 10)])
    scale_add = torch.tensor([0.0625, 0.0625, 10, 0])

    return torch.mul(X, scale_mult) + scale_add
