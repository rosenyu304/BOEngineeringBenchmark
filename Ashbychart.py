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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
##########################################################################################
##########################################################################################
# Functions
##########################################################################################
##########################################################################################



def Ashbychart(individuals): # This should be the test function
    fx = torch.zeros((individuals.shape[0],1))
    gA = torch.zeros((individuals.shape[0],1))

    Pcr = 10
    L = 2
    C = 1
    
    for ii in range(individuals.shape[0]):
        x = individuals[ii,:]
        rho = x[0]
        Sy   = x[1]
        E   = x[2]
        fx[ii] = (- rho*L*2*Pcr/Sy )
        gA[ii] = (3*L**2 * Sy**2) / (C*np.pi*np.pi*Pcr) - E
        
    # fx = torch.tensor(fx)
    # fx = torch.reshape(fx, (len(fx),1))
    # gA = torch.tensor(gA)
    # gA = torch.reshape(gA, (len(fx),1))

    return gA, fx




def Ashbychart_Scaling(X): # This should be the test function

    l_bounds = [10, 0.01, 1e-4 ]
    u_bounds = [50000,10000,1000]
    X_scaled = qmc.scale(X, l_bounds, u_bounds)

    return X_scaled


def Ashbychart_load_pixelX_for_gc(X):
    
    # print(X)
    
    rho = (X[:,0]  * (np.log10(50000)-np.log10(10  )) + np.log10(10    )   ).reshape(X.shape[0],1)
    Sy  = (X[:,1]  * (np.log10(10000)-np.log10(0.01  )) + np.log10(0.01  )   ).reshape(X.shape[0],1)
    E   = (X[:,2]  * (np.log10(1000 )-np.log10(1e-4)) + np.log10(1e-4  )   ).reshape(X.shape[0],1)
    
    rho = torch.tensor(rho).reshape(X.shape[0],1)
    Sy = torch.tensor(Sy).reshape(X.shape[0],1)
    E = torch.tensor(E).reshape(X.shape[0],1)
    

    X_unnormed = torch.cat((rho, Sy, E), dim=1)
    
    # First Ashby Chart
    left_bound = 215
    right_bound = 795
    bottom_bound = 95
    up_bound = 472

    rhoE_x_scale = (np.log10(50000)-np.log10(10)) / (right_bound-left_bound)
    rhoE_y_scale = (np.log10(1000)-np.log10(1e-4)) / (up_bound-bottom_bound)

    log_rho = torch.round( ( rho - np.log10(10) )/rhoE_x_scale + left_bound )
    log_E = torch.round( up_bound- ( E -np.log10(1e-4) )/rhoE_y_scale )

    # Second Ashby Chart
    left_bound = 215
    right_bound = 795
    bottom_bound = 55
    up_bound = 472
    rhoSy_y_scale = (np.log10(10000)-np.log10(1e-2)) / (up_bound-bottom_bound)

    log_Sy =  torch.round( up_bound- ( Sy - np.log10(1e-2) )/rhoSy_y_scale )

    pixel_X = torch.cat((log_rho, log_Sy, log_E), dim=1)
    
    return pixel_X



def Ashbychart_get_gc(pixel_X):
 
    # Read Images
    SyvsRho_gray_img = mpimg.imread('test_functions/SyvsRho_gray.png')
    EvsRho_gray_img = mpimg.imread('test_functions/EvsRho_gray.png')
    
    gc = torch.zeros(pixel_X.shape[0], 1)
    ind = 0
    
    for x in pixel_X:
        rho = int(x[0])
        Sy  = int(x[1])
        E   = int(x[2])
        if (SyvsRho_gray_img[Sy,rho,1] != 1)  & (EvsRho_gray_img[E,rho,1] != 1) :
            gc[ind]=1
        else:
            gc[ind]=-1
        ind+=1
    
    
    return gc




























