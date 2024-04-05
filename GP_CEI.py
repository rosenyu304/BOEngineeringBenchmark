import torch
import numpy as np

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.optim import optimize_acqf


def get_and_fit_gp(X, Y):
    gp = SingleTaskGP(X, Y) 
    mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
    fit_gpytorch_mll(mll)
    return gp


def get_next_candidates_CEI(X, gx, fx, best_f):
    
    # 1. Normalized gx and fx => Get Y by concate gx and fx
    gx_norm = gx
    for ii in range(gx.shape[1]):
        gx_norm[:,ii] = gx_norm[:,ii] / (torch.max(gx_norm[:,ii]) - torch.min(gx_norm[:,ii]))

    fx_norm = fx
    fx_norm = fx / (torch.max(fx) - torch.min(fx))
    
    Normed_Y = torch.cat((gx_norm,fx_norm),dim=1)


    
    

    # 2. Fit GP model
    model = get_and_fit_gp(X.double(),Normed_Y.double())

    

    # 3.Get Constraints
    constraints = {x: (None, 0.0) for x in range(gx.shape[1])}

    

    # 4. Optimized ACQ
    obj_index = gx.shape[1]
    bounds = torch.cat((torch.zeros(1,X.shape[1]), torch.ones(1,X.shape[1])))
    cEI = ConstrainedExpectedImprovement(model, best_f, obj_index, constraints)  
    best_candidate, best_eci_value = optimize_acqf(
        acq_function=cEI,
        bounds=bounds,
        q=1,
        num_restarts=1,
        raw_samples=3, 
    )

    
    
    return best_candidate



