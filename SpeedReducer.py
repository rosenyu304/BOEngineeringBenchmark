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


def SpeedReducer(individuals): # This should be the test function
    
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
        
        b = x[0]
        m = x[1]
        z = x[2]
        L1 = x[3]
        L2 = x[4]
        d1 = x[5]
        d2 = x[6]
        
        C1 = 0.7854*b*m*m
        C2 = 3.3333*z*z + 14.9334*z - 43.0934
        C3 = 1.508*b*(d1*d1 + d2*d2)
        C4 = 7.4777*(d1*d1*d1 + d2*d2*d2)
        C5 = 0.7854*(L1*d1*d1 + L2*d2*d2)
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( 0.7854*b*m*m * (3.3333*z*z + 14.9334*z - 43.0934) - 1.508*b*(d1*d1 + d2*d2) + 7.4777*(d1*d1*d1 + d2*d2*d2) + 0.7854*(L1*d1*d1 + L2*d2*d2)  )

        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 27/(b*m*m*z) - 1
        g2 = 397.5/(b*m*m*z*z) - 1
        
        
        g3 = 1.93*L1**3 /(m*z *d1**4) - 1
        g4 = 1.93*L2**3 /(m*z *d2**4) - 1
        
        
        
        g5 = np.sqrt( (745*L1/(m*z))**2 + 1.69*1e6 ) / (110*d1**3) -1
        g6 = np.sqrt( (745*L2/(m*z))**2 + 157.5*1e6 ) / (85*d2**3) -1
        g7 = m*z/40 - 1
        g8 = 5*m/(b) - 1
        g9 = b/(12*m) -1

        
        
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

    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5, gx6, gx7, gx8, gx9), 1)

    #############################################################################
    #############################################################################
    
    
    return gx, fx



def SpeedReducer_Scaling(X):

    b  = (X[:,0] * ( 3.6 - 2.6 ) + 2.6).reshape(X.shape[0],1)
    m  = (X[:,1] * ( 0.8 - 0.7 ) + 0.7).reshape(X.shape[0],1)
    z  = (X[:,2] * ( 28 - 17 ) + 17).reshape(X.shape[0],1)
    L1 = (X[:,3] * ( 8.3 - 7.3 ) + 7.3).reshape(X.shape[0],1)
    L2 = (X[:,4] * ( 8.3 - 7.3 ) + 7.3).reshape(X.shape[0],1)
    d1 = (X[:,5] * ( 3.9 - 2.9 ) + 2.9).reshape(X.shape[0],1)
    d2 = (X[:,6] * ( 5.5 - 5 ) + 5).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((b, m, z, L1, L2, d1, d2), dim=1)
    return X_scaled


















































