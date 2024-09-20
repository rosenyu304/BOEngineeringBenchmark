import torch
import numpy as np

#
#
#   Car: 11D objective, 10 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-24):2325–2336
#
#

def Car(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 11, "Input must be an n-by-11 PyTorch tensor."

    n = individuals.size(0)

    fx, gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11 = (
                    torch.zeros((n,1)), torch.zeros((n,1)), torch.zeros((n,1)),
                    torch.zeros((n,1)), torch.zeros((n,1)), torch.zeros((n,1)),
                    torch.zeros((n,1)), torch.zeros((n,1)), torch.zeros((n,1)),
                    torch.zeros((n,1)), torch.zeros((n,1)), torch.zeros((n,1)))

    # Set function and constraints here:
    for i in range(n):

        x = individuals[i,:]

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x

        ## Negative sign to make it a maximization problem
        test_function = - ( 1.98 + 4.90*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 2.73*x7 )

        ## Calculate constraints terms
        gx1[i] = 1.16 - 0.3717*x2*x4 - 0.00931*x2*x10 - 0.484*x3*x9 + 0.01343*x6*x10   -1

        gx2[i] = (0.261 - 0.0159*x1*x2 - 0.188*x1*x8
              - 0.019*x2*x7 + 0.0144*x3*x5 + 0.0008757*x5*x10
              + 0.08045*x6*x9 + 0.00139*x8*x11 + 0.00001575*x10*x11)        -0.9

        gx3[i] = (0.214 + 0.00817*x5 - 0.131*x1*x8 - 0.0704*x1*x9 + 0.03099*x2*x6
              -0.018*x2*x7 + 0.0208*x3*x8 + 0.121*x3*x9 - 0.00364*x5*x6
              +0.0007715*x5*x10 - 0.0005354*x6*x10 + 0.00121*x8*x11)        -0.9

        gx4[i] = 0.74 -0.061*x2 -0.163*x3*x8 +0.001232*x3*x10 -0.166*x7*x9 +0.227*x2*x2        -0.9

        gx5[i] = 28.98 +3.818*x3-4.2*x1*x2+0.0207*x5*x10+6.63*x6*x9-7.7*x7*x8+0.32*x9*x10    -32

        gx6[i] = 33.86 +2.95*x3+0.1792*x10-5.057*x1*x2-11.0*x2*x8-0.0215*x5*x10-9.98*x7*x8+22.0*x8*x9    -32

        gx7[i] = 46.36 -9.9*x2-12.9*x1*x8+0.1107*x3*x10    -32

        gx8[i] = 4.72 -0.5*x4-0.19*x2*x3-0.0122*x4*x10+0.009325*x6*x10+0.000191*x11**2     -4

        gx9[i] = 10.58 -0.674*x1*x2-1.95*x2*x8+0.02054*x3*x10-0.0198*x4*x10+0.028*x6*x10     -9.9

        gx10[i] = 16.45 -0.489*x3*x7-0.843*x5*x6+0.0432*x9*x10-0.0556*x9*x11-0.000786*x11**2     -15.7

        fx[i] = test_function

    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10), 1)
    return gx, fx

def Car_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 11, "Input must be an n-by-11 PyTorch tensor."

    scale_mult = torch.tensor([(1.5-0.5), (1.35-0.45), (1.5-0.5), (1.5-0.5), (1.5-0.5),
                               (1.5-0.5), (1.5-0.5), (0.345-0.192), (0.345-0.192), (-20), (-20)])
    scale_add = torch.tensor([0.5, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.192, 0.192, 0, 0])

    return torch.mul(X, scale_mult) + scale_add
