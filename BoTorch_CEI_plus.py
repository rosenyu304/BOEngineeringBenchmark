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
from botorch.acquisition.analytic import ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _preprocess_constraint_bounds, _scaled_improvement, _ei_helper

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

# from __future__ import annotations

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

#####################################################################################################
#####################################################################################################
# CEI_MIN Setup
#####################################################################################################
#####################################################################################################
#####################################################################################################

class ConstrainedExpectedImprovement_plus(AnalyticAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        maximize: bool = True,
    ) -> None:

        # Use AcquisitionFunction constructor to avoid check for posterior transform.
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.maximize = maximize
        self.objective_index = objective_index
        self.constraints = constraints
        self.register_buffer("best_f", torch.as_tensor(best_f))
        _preprocess_constraint_bounds(self, constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:

        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        ei = sigma_obj * _ei_helper(u)
        log_prob_feas = compute_log_prob_feas(self, means=means, sigmas=sigmas)
        prob_f = log_prob_feas.exp()

        return ei.mul(log_prob_feas.exp())


    
def compute_log_prob_feas(
    acqf: Union[LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement],
    means: Tensor,
    sigmas: Tensor,
) -> Tensor:

    acqf.to(device=means.device)
    log_prob = torch.zeros_like(means[..., 0])
    if len(acqf.con_lower_inds) > 0:
        i = acqf.con_lower_inds
        dist_l = (acqf.con_lower - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_Phi(-dist_l).sum(dim=-1)  # 1 - Phi(x) = Phi(-x)
    if len(acqf.con_upper_inds) > 0:
        i = acqf.con_upper_inds
        dist_u = (acqf.con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_Phi(dist_u).sum(dim=-1)
    if len(acqf.con_both_inds) > 0:
        i = acqf.con_both_inds
        con_lower, con_upper = acqf.con_both[:, 0], acqf.con_both[:, 1]
        # scaled distance to lower and upper constraint boundary:
        dist_l = (con_lower - means[..., i]) / sigmas[..., i]
        dist_u = (con_upper - means[..., i]) / sigmas[..., i]
        log_prob = log_prob + log_prob_normal_in(a=dist_l, b=dist_u).sum(dim=-1)
    
    # 𝑚𝑖𝑛(1, 2𝑃𝐹𝑖(𝑥))
    for ii in range(log_prob.size(-1)):
        if log_prob[ii] + np.log(2) > np.log(1):
            log_prob[ii]=np.log(1)
    
    return log_prob