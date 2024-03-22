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


def WeldedBeam(individuals): # This should be the test function
    
    
    C1 = 1.10471
    C2 = 0.04811
    C3 = 14.0
    fx = torch.zeros(individuals.shape[0], 1)
    gx1 = torch.zeros(individuals.shape[0], 1)
    gx2 = torch.zeros(individuals.shape[0], 1)
    gx3 = torch.zeros(individuals.shape[0], 1)
    gx4 = torch.zeros(individuals.shape[0], 1)
    gx5 = torch.zeros(individuals.shape[0], 1)
    
    for i in range(individuals.shape[0]):
        
        x = individuals[i,:]

        h = x[0]
        l = x[1]
        t = x[2]
        b = x[3]
        
        test_function = - ( C1*h*h*l + C2*t*b*(C3+l) )
        fx[i] = test_function
        
        ## Calculate constraints terms 
        tao_dx = 6000 / (np.sqrt(2)*h*l)
        
        tao_dxx = 6000*(14+0.5*l)*np.sqrt( 0.25*(l**2 + (h+t)**2 ) ) / (2* (0.707*h*l * ( l**2 /12 + 0.25*(h+t)**2 ) ) )
        
        tao = np.sqrt( tao_dx**2 + tao_dxx**2 + l*tao_dx*tao_dxx / np.sqrt(0.25*(l**2 + (h+t)**2)) )
        
        sigma = 504000/ (t**2 * b)
        
        P_c = 64746*(1-0.0282346*t)* t * b**3
        
        delta = 2.1952/ (t**3 *b)
        
        
        ## Calculate 5 constraints
        g1 = (-1) * (13600- tao) 
        g2 = (-1) * (30000 - sigma) 
        g3 = (-1) * (b - h)
        g4 = (-1) * (P_c - 6000) 
        g5 = (-1) * (0.25 - delta)
        
        gx1[i] =  g1        
        gx2[i] =  g2     
        gx3[i] =  g3             
        gx4[i] =  g4 
        gx5[i] =  g5 
    
    
    
    gx = torch.cat((gx1, gx2, gx3, gx4, gx5), 1)
    return gx, fx





def WeldedBeam_Scaling(X):
    h = (X[:,0]  * (10-0.125) + 0.125 ).reshape(X.shape[0],1)
    l = (X[:,1]  * (15-0.1  ) + 0.1   ).reshape(X.shape[0],1)
    t = (X[:,2]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    b = (X[:,3]  * (10-0.1 ) + 0.1         ).reshape(X.shape[0],1)
    
    X_scaled = torch.cat((h, l, t, b), dim=1)
    return X_scaled








