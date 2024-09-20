import torch
import numpy as np

#
#
#   WeldedBeam: 4D objective, 5 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-24):2325–2336
#
#

def WeldedBeam(individuals):

    assert torch.is_tensor(individuals) and individuals.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."

    n = individuals.shape[0]

    C1, C2, C3 = (1.10471, 0.04811, 14.0)
    fx, gx1, gx2, gx3, gx4, gx5 = (torch.zeros(n, 1), torch.zeros(n, 1),
                              torch.zeros(n, 1), torch.zeros(n, 1),
                              torch.zeros(n, 1), torch.zeros(n, 1))

    for i in range(n):

        x = individuals[i,:]

        h, l, t, b = x

        test_function = - ( C1*h*h*l + C2*t*b*(C3+l) )
        fx[i] = test_function

        ## Calculate constraints terms
        tao_dx = 6000 / (np.sqrt(2)*h*l)

        tao_dxx = 6000*(14+0.5*l)*np.sqrt( 0.25*(l**2 + (h+t)**2 ) ) / (2* (0.707*h*l * ( l**2 /12 + 0.25*(h+t)**2 ) ) )

        tao = np.sqrt( tao_dx**2 + tao_dxx**2 + l*tao_dx*tao_dxx / np.sqrt(0.25*(l**2 + (h+t)**2)) )

        sigma = 504000/ (t**2 * b)

        P_c = 64746*(1-0.0282346*t)* t * b**3

        delta = 2.1952/ (t**3 *b)

        ## Calculate the 5 constraints
        gx1[i] =  (-1) * (13600- tao)
        gx2[i] =  (-1) * (30000 - sigma)
        gx3[i] =  (-1) * (b - h)
        gx4[i] =  (-1) * (P_c - 6000)
        gx5[i] =  (-1) * (0.25 - delta)

    gx = torch.cat((gx1, gx2, gx3, gx4, gx5), 1)
    return gx, fx


def WeldedBeam_Scaling(X):

    assert torch.is_tensor(X) and X.size(1) == 4, "Input must be an n-by-4 PyTorch tensor."

    scale_mult = torch.tensor([(10-0.125), (15-0.1), (10-0.1), (10-0.1)])
    scale_add = torch.tensor([0.125, 0.1, 0.1, 0.1])

    return torch.mul(X, scale_mult) + scale_add
