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





def CantileverBeam(individuals): # This should be the test function
    
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    gx5 = []
    gx6 = []
    gx7 = []
    gx8 = []
    gx9 = []
    gx10 = []
    gx11 = []
    
    

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
        
        P = 50000
        E = 2*107
        L = 100
        
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( x1*x6*L + x2*x7*L + x3*x8*L + x4*x9*L + x5*x10*L )
        # test_function = - ( C1*C2 - C3 + C4 + C5 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 600 * P / (x5*x10*x10) - 14000
        g2 = 6 * P * (L*2) / (x4*x9*x9) - 14000
        g3 = 6 * P * (L*3) / (x3*x8*x8) - 14000
        g4 = 6 * P * (L*4) / (x2*x7*x7) - 14000
        g5 = 6 * P * (L*5) / (x1*x6*x6) - 14000
        g6 = P* L**3 * (1/L + 7/L + 19/L + 37/L + 61/L) / (3*E) -2.7
        g7 = x10/x5 - 20
        g8 = x9/x4 - 20
        g9 = x8/x3 - 20
        g10 = x7/x2 - 20
        g11 = x6/x1 - 20

        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
        gx7.append( g7 )            
        gx8.append( g8 )
        gx9.append( g9 )
        gx10.append( g10 )
        gx11.append( g11 )
    
    # print(gx5)
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    gx1 = torch.tensor(gx1)  
    gx1 = gx1.reshape((n, 1))

    gx2 = torch.tensor(gx2)  
    gx2 = gx2.reshape((n, 1))
    
    gx3 = torch.tensor(gx3)  
    gx3 = gx3.reshape((n, 1))
    
    gx4 = torch.tensor(gx4)  
    gx4 = gx4.reshape((n, 1))
    
    gx5 = torch.tensor(gx5)  
    gx5 = gx1.reshape((n, 1))

    gx6 = torch.tensor(gx6)  
    gx6 = gx2.reshape((n, 1))
    
    gx7 = torch.tensor(gx7)  
    gx7 = gx3.reshape((n, 1))
    
    gx8 = torch.tensor(gx8)  
    gx8 = gx4.reshape((n, 1))
    
    gx9 = torch.tensor(gx9)  
    gx9 = gx4.reshape((n, 1))
    
    gx10 = torch.tensor(gx10)  
    gx10 = gx4.reshape((n, 1))
    
    gx11 = torch.tensor(gx11)  
    gx11 = gx4.reshape((n, 1))
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9, gx10, gx11), 1)
    #############################################################################
    #############################################################################
    
    
    return gx, fx




def CantileverBeam_Scaling(X):
    x1 = (X[:,0] * (5-1) + 1).reshape(X.shape[0],1)
    x2 = (X[:,1] * (5-1) + 1).reshape(X.shape[0],1)
    x3 = (X[:,2] * (5-1) + 1).reshape(X.shape[0],1)
    x4 = (X[:,3] * (5-1) + 1).reshape(X.shape[0],1)
    x5 = (X[:,4] * (5-1) + 1).reshape(X.shape[0],1)
    x6 = (X[:,5] * (65-30) + 30).reshape(X.shape[0],1)
    x7 = (X[:,6] * (65-30) + 30).reshape(X.shape[0],1)
    x8 = (X[:,7] * (65-30) + 30).reshape(X.shape[0],1)
    x9 = (X[:,8] * (65-30) + 30).reshape(X.shape[0],1)
    x10 = (X[:,9] * (65-30) + 30).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), dim=1)
    return X_scaled































