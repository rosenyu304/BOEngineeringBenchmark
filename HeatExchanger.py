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


def HeatExchanger(individuals): # This should be the test function
    
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
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( x1+x2+x3 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 0.0025 * (x4+x6) - 1
        g2 = 0.0025 * (x5 + x7 - x4) - 1
        g3 = 0.01 *(x8-x5) - 1
        g4 = 833.33252*x4 + 100*x1 - x1*x6 - 83333.333
        g5 = 1250*x5 + x2*x4 - x2*x7 - 125*x4
        g6 = x3*x5 - 2500*x5 - x3*x8 + 125*10000


        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
        gx5.append( g5 )       
        gx6.append( g6 )    
    
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
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6), 1)

    #############################################################################
    #############################################################################

    return gx, fx





def HeatExchanger_Scaling(X):
    
    x1 = (X[:,0] * (10000-100) + 100).reshape(X.shape[0],1)
    x2 = (X[:,1] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x3 = (X[:,2] * (10000-1000) + 1000).reshape(X.shape[0],1)
    x4 = (X[:,3] * (1000-10) + 10).reshape(X.shape[0],1)
    x5 = (X[:,4] * (1000-10) + 10).reshape(X.shape[0],1)
    x6 = (X[:,5] * (1000-10) + 10).reshape(X.shape[0],1)
    x7 = (X[:,6] * (1000-10) + 10).reshape(X.shape[0],1)
    x8 = (X[:,7] * (1000-10) + 10).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)
    
    return X_scaled







































































