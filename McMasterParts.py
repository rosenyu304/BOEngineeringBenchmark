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


def McMasterParts(individuals): # This should be the test function
    fx = []
    
    for x in individuals:
        L = x[0]
        H = x[1]
        test_function = (- L*H ) 
        fx.append(test_function) 
    fx = torch.tensor(fx)
    fx = torch.reshape(fx, (len(fx),1))

    return fx


def McMasterParts_Scaling(X):
    X_Scaled = X*80+20
    return X_Scaled








def McMasterParts_get_GC_label(individuals, McMaster_Array):
    
    # User set this: how far do you want the sample point to be close to the McMaster gear?
    radii = 3
    
    # Label feasible data: 1 (Not feasible: -1)
    XX = individuals
    condd_matrix = torch.zeros(XX.shape[0],36)
    for ii in range(36):
        condd = ((XX[:,0] >= (McMaster_Array[ii,0] - radii)) & (XX[:,0] <= (McMaster_Array[ii,0] + radii) ) &
            (XX[:,1] >= (McMaster_Array[ii,1] - radii)) & (XX[:,1] <= (McMaster_Array[ii,1] + radii) ))
        condd_matrix[:,ii]=condd
    gc = torch.sum(condd_matrix, dim=1)
    gc[gc==0]=-1
    
    
    gc = torch.tensor(gc)
    gc = torch.reshape(gc, (len(gc),1))
    return gc





def McMaster_data():
    McMaster_A          = np.array([22,27,25,31,45,40])
    McMaster_B          = np.array([28,36,32,40,60,55])
    McMaster_Gear_Dia   = np.array([25,32,25,30,40,40])
    McMaster_Pinion_Dia = np.array([28,36,32,40,60,55])

    McMaster_Heights = np.zeros((6,1))
    McMaster_Widths  = np.zeros((6,1))
    McMaster_Array = np.zeros((36,2))

    for ii in range(6):
        AA = McMaster_A[ii]
        PD = McMaster_Pinion_Dia[ii]
        McMaster_Heights[ii]=AA+PD/2

    for ii in range(6):
        BB = McMaster_B[ii]
        GD = McMaster_Gear_Dia[ii]
        McMaster_Widths[ii]=BB+GD/2

    for ii in range(6):
        for jj in range(6):
            McMaster_Array[ii+6*jj,0]=McMaster_Widths[ii]
            McMaster_Array[ii+6*jj,1]=McMaster_Heights[jj]
            
    return McMaster_Array






















