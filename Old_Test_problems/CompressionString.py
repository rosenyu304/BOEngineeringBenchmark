from __future__ import annotations

import os
import torch
import numpy as np
import plotly
import plotly.graph_objects as go
import warnings
import time
import matplotlib.pyplot as plt
import math




from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _preprocess_constraint_bounds, _scaled_improvement, _ei_helper, _compute_log_prob_feas, _log_ei_helper


warnings.filterwarnings('ignore')

device = torch.device("cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

NOISE_SE = 0
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)


N_DIMENSION = 2


from abc import ABC

from contextlib import nullcontext
from copy import deepcopy

from typing import Dict, Optional, Tuple, Union

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


from botorch.test_functions import Ackley

import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping


#####################################################################################################
#####################################################################################################
# Functions
#####################################################################################################
#####################################################################################################
#####################################################################################################


def CompressionString(individuals): # This should be the test function
    
    
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
        # print(x)
        
        d = x[0]
        D = x[1]
        N = x[2]
        
        
        ## Negative sign to make it a maximization problem
        test_function = - ( (N+2)*D*d**2 )
        fx.append(test_function) 
        
        ## Calculate constraints terms 
        g1 = 1 -  ( D*D*D * N / (71785* d*d*d*d) )
        g2 = (4*D*D - D*d) / (12566 * (D*d*d*d - d*d*d*d)) + 1/(5108*d*d) -  1
        g3 = 1 - 140.45*d / (D*D * N)
        g4 = (D+d)/1.5 - 1
        
        
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



def CompressionString_Scaling(X): 
    
    d = (X[:,0] * ( 1   - 0.05 ) + 0.05 ).reshape(X.shape[0],1)
    D = (X[:,1] * ( 1.3 - 0.25 ) + 0.25   ).reshape(X.shape[0],1)
    N = (X[:,2]  * ( 15  - 2    ) + 2         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((d, D, N), dim=1)

    return X_scaled








