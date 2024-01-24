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


def KeaneBump30D(X): 
    
    fx = torch.zeros(X.shape[0], 1)
    gx1 = torch.zeros(X.shape[0], 1)
    gx2 = torch.zeros(X.shape[0], 1)
    
    
    
    for i in range(X.shape[0]):
        x = X[i,:]
        
        cos4 = 0
        cos2 = 1
        sq_denom = 0
        
        pi_sum = 1
        sigma_sum = 0
        
        for j in range(X.shape[1]):
            cos4 += torch.cos(x[j]) ** 4
            cos2 *= torch.cos(x[j]) ** 2
            sq_denom += (j+1) * (x[j])**2
            
            pi_sum *= x[j]
            sigma_sum += x[j]
        
        
        # Objective
        test_function = torch.abs(  (cos4 - 2*cos2) / torch.sqrt(sq_denom)  )
        fx[i] = test_function

        # Constraint
        gx1[i] = 0.75 - pi_sum
        gx2[i] = sigma_sum - 7.5* (X.shape[1])
        
    gx = torch.cat((gx1, gx2), 1)
    return gx, fx





def KeaneBump30D_Scaling(X):
    
    X_scaled = X*10
    
    return X_scaled





















