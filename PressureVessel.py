
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




def PressureVessel(individuals): # This should be the test function
    
    C1 = 0.6224
    C2 = 1.7781
    C3 = 3.1661
    C4 = 19.84
    fx = []
    gx1 = []
    gx2 = []
    gx3 = []
    gx4 = []
    

    #############################################################################
    #############################################################################
    n = individuals.size(0)
    
    # Set function and constraints here:
    for i in range(n):
        
        x = individuals[i,:]
        
        Ts = x[0]
        Th = x[1]
        R = x[2]
        L = x[3]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( C1*Ts*R*L + C2*Th*R*R + C3*Ts*Ts*L + C4*Ts*Ts*R )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = -Ts + 0.0193*R
        g2 = -Th + 0.00954*R
        g3 = (-1)*np.pi*R*R*L + (-1)*4/3*np.pi*R*R*R + 750*1728
        
        g4 = L-240
        
        
        ## Calculate 5 constraints
        gx1.append( g1 )       
        gx2.append( g2 )    
        gx3.append( g3 )            
        gx4.append( g4 )
       
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
    
    gx = torch.cat((gx1, gx2, gx3, gx4), 1)
    #############################################################################
    #############################################################################
    
    return gx, fx






def PressureVessel_Scaling(X):
    
    Ts  = (X[:,0] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    Th  = (X[:,1] * (98*0.0625) + 0.0625).reshape(X.shape[0],1)
    R   = (X[:,2] * (200-10) + 10).reshape(X.shape[0],1)
    L   = (X[:,3] * (200-10) ).reshape(X.shape[0],1)
    
    
    X_scaled = torch.cat((Ts, Th, R, L), dim=1)
    
    return X_scaled





































































