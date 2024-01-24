from __future__ import annotations
import os
import torch
import numpy as np
import plotly
import plotly.graph_objects as go
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _preprocess_constraint_bounds, _scaled_improvement, _ei_helper, _compute_log_prob_feas, _log_ei_helper

import numpy as np
import matplotlib.pyplot as plt

import warnings
import time

warnings.filterwarnings('ignore')

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

NOISE_SE = 0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)


N_DIMENSION = 2

import math

from abc import ABC

from contextlib import nullcontext
from copy import deepcopy

from typing import Dict, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.constants import get_constants_like
from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi,
    log_prob_normal_in,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp, logmeanexp
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from torch.nn.functional import pad

# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

import warnings

import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

warnings.filterwarnings('ignore')

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

NOISE_SE = 0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)


N_DIMENSION = 2

import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping

device = 'cpu:0'


##########################################################################################
##########################################################################################
# Functions
##########################################################################################
##########################################################################################

def Car(individuals): # This should be the test function
    
    n = individuals.size(0)
    
    fx = torch.zeros((n,1))
    gx1 = torch.zeros((n,1))
    gx2 = torch.zeros((n,1))
    gx3 = torch.zeros((n,1))
    gx4 = torch.zeros((n,1))
    gx5 = torch.zeros((n,1))
    gx6 = torch.zeros((n,1))
    gx7 = torch.zeros((n,1))
    gx8 = torch.zeros((n,1))
    gx9 = torch.zeros((n,1))
    gx10 = torch.zeros((n,1))
    gx11 = torch.zeros((n,1))
    
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        # print(x)
        
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]

        
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 1.98 + 4.90*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 2.73*x7 )
        
        ## Calculate constraints terms 
        g1 = 1.16 - 0.3717*x2*x4 - 0.00931*x2*x10 - 0.484*x3*x9 + 0.01343*x6*x10   -1
        
        g2 = (0.261 - 0.0159*x1*x2 - 0.188*x1*x8 
              - 0.019*x2*x7 + 0.0144*x3*x5 + 0.0008757*x5*x10
              + 0.08045*x6*x9 + 0.00139*x8*x11 + 0.00001575*x10*x11)        -0.9
        
        g3 = (0.214 + 0.00817*x5 - 0.131*x1*x8 - 0.0704*x1*x9 + 0.03099*x2*x6
              -0.018*x2*x7 + 0.0208*x3*x8 + 0.121*x3*x9 - 0.00364*x5*x6
              +0.0007715*x5*x10 - 0.0005354*x6*x10 + 0.00121*x8*x11)        -0.9
        
        g4 = 0.74 -0.061*x2 -0.163*x3*x8 +0.001232*x3*x10 -0.166*x7*x9 +0.227*x2*x2        -0.9
        
        g5 = 28.98 +3.818*x3-4.2*x1*x2+0.0207*x5*x10+6.63*x6*x9-7.7*x7*x8+0.32*x9*x10    -32
        
        g6 = 33.86 +2.95*x3+0.1792*x10-5.057*x1*x2-11.0*x2*x8-0.0215*x5*x10-9.98*x7*x8+22.0*x8*x9    -32
        
        g7 = 46.36 -9.9*x2-12.9*x1*x8+0.1107*x3*x10    -32
        
        g8 = 4.72 -0.5*x4-0.19*x2*x3-0.0122*x4*x10+0.009325*x6*x10+0.000191*x11**2     -4
        
        g9 = 10.58 -0.674*x1*x2-1.95*x2*x8+0.02054*x3*x10-0.0198*x4*x10+0.028*x6*x10     -9.9
        
        g10 = 16.45 -0.489*x3*x7-0.843*x5*x6+0.0432*x9*x10-0.0556*x9*x11-0.000786*x11**2     -15.7
        
        
        gx1[i] = g1
        gx2[i] = g2
        gx3[i] = g3
        gx4[i] = g4
        gx5[i] = g5
        gx6[i] = g6
        gx7[i] = g7
        gx8[i] = g8
        gx9[i] = g9
        gx10[i] = g10
        fx[i] = test_function
        
        
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10), 1)
    #############################################################################
    #############################################################################

    
    return gx, fx




def Car_Scaling(X):
    x1  = (X[:,0] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x2  = (X[:,1] * (1.35-0.45) + 0.45).reshape(X.shape[0],1)
    x3  = (X[:,2] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x4 = (X[:,3] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x5 = (X[:,4] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x6 = (X[:,5] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x7 = (X[:,6] * (1.5-0.5) + 0.5).reshape(X.shape[0],1)
    x8 = (X[:,7] * (0.345-0.192) + 0.192).reshape(X.shape[0],1)
    x9 = (X[:,8] * (0.345-0.192) + 0.192).reshape(X.shape[0],1)
    x10 = (X[:,9] * (-20)).reshape(X.shape[0],1)
    x11 = (X[:,10] * (-20)).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), dim=1)
    return X_scaled





























