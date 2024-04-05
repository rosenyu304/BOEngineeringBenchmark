import torch
import numpy as np


from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf


def get_and_fit_gp(X, Y):
    gp = SingleTaskGP(X, Y) 
    mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
    fit_gpytorch_mll(mll)
    return gp

def get_next_candidates(X, gx, fx, best_f):

    ####################################################################################
    # 1. Normalized Y
    Y = fx
    Normed_Y = Y/(torch.max(Y)-torch.min(Y))
    ####################################################################################
    
    ####################################################################################
    # 2. Fit GP model
    model = get_and_fit_gp(X.double(),Normed_Y.double())
    ####################################################################################
    
    ####################################################################################
    # 3. Optimized ACQ
    bounds = torch.cat((torch.zeros(1,X.shape[1]), torch.ones(1,X.shape[1])))
    EI = ExpectedImprovement(model, best_f)  
    best_candidate, best_eci_value = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=1,
        num_restarts=1,
        raw_samples=3, 
    )
    ####################################################################################
    
    
    return best_candidate