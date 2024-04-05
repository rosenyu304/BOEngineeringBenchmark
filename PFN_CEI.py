import contextlib
import torch
import scipy
import math
from sklearn.preprocessing import power_transform, PowerTransformer

from torchvision.transforms.functional import to_tensor
from pfns4bo import transformer
from pfns4bo import bar_distribution





################################################################################
## PFN defined functions
################################################################################


def log01(x, eps=.0000001, input_between_zero_and_one=False):
    logx = torch.log(x + eps)
    if input_between_zero_and_one:
        return (logx - math.log(eps)) / (math.log(1 + eps) - math.log(eps))
    return (logx - logx.min(0)[0]) / (logx.max(0)[0] - logx.min(0)[0])

def log01_batch(x, eps=.0000001, input_between_zero_and_one=False):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        x[:, b, b] = log01(x[:, b, b], eps=eps, input_between_zero_and_one=input_between_zero_and_one)
    return x

def lognormed_batch(x, eval_pos, eps=.0000001):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        logx = torch.log(x[:, b, b]+eps)
        x[:, b, b] = (logx - logx[:eval_pos].mean(0))/logx[:eval_pos].std(0)
    return x

def _rank_transform(x_train, x):
    assert len(x_train.shape) == len(x.shape) == 1
    relative_to = torch.cat((torch.zeros_like(x_train[:1]),x_train.unique(sorted=True,), torch.ones_like(x_train[-1:])),-1)
    higher_comparison = (relative_to < x[...,None]).sum(-1).clamp(min=1)
    pos_inside_interval = (x - relative_to[higher_comparison-1])/(relative_to[higher_comparison] - relative_to[higher_comparison-1])
    x_transformed = higher_comparison - 1 + pos_inside_interval
    return x_transformed/(len(relative_to)-1.)

def rank_transform(x_train, x):
    assert x.shape[1] == x_train.shape[1], f"{x.shape=} and {x_train.shape=}"
    # make sure everything is between 0 and 1
    assert (x_train >= 0.).all() and (x_train <= 1.).all(), f"{x_train=}"
    assert (x >= 0.).all() and (x <= 1.).all(), f"{x=}"
    return_x = x.clone()
    for feature_dim in range(x.shape[1]):
        return_x[:, feature_dim] = _rank_transform(x_train[:, feature_dim], x[:, feature_dim])
    return return_x



def general_power_transform(x_train, x_apply, eps, less_safe=False):
    if eps > 0:
        try:
            pt = PowerTransformer(method='box-cox')
            pt.fit(x_train.cpu()+eps)
            x_out = torch.tensor(pt.transform(x_apply.cpu()+eps), dtype=x_apply.dtype, device=x_apply.device)
        except Exception as e:
            print(e)
            x_out = x_apply - x_train.mean(0)
            print(x_train)
            print(x_out)
    else:
        pt = PowerTransformer(method='yeo-johnson')
        if not less_safe and (x_train.std() > 1_000 or x_train.mean().abs() > 1_000):
            x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            x_train = (x_train - x_train.mean(0)) / x_train.std(0)
            # print('inputs are LAARGEe, normalizing them')
        try:
            pt.fit(x_train.cpu().double())
        # except ValueError as e:
        except Exception as e:
            print(x_train)
            print('caught this errrr', e)
            if less_safe:
                x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            else:
                x_train = x_train - x_train.mean(0)
                x_apply = x_apply - x_train.mean(0)
            print(x_train)
            pt.fit(x_train.cpu().double())
            print(x_train)
        x_out = torch.tensor(pt.transform(x_apply.cpu()), dtype=x_apply.dtype, device=x_apply.device)
    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        print('WARNING: power transform failed')
        print(f"{x_train=} and {x_apply=}")
        x_out = x_apply - x_train.mean(0)
    return x_out


def acq_ensembling(acq_values): # (points, ensemble dim)
    return acq_values.max(1).values

################################################################################
## End of PFN defined functions 
################################################################################




################################################################################
## PFNCEI:
################################################################################



def PFNCEI_general_acq_function(model: transformer.TransformerModel, x_given, y_given, x_eval,  GX, apply_power_transform=True,
                    rand_sample=False, znormalize=False, pre_normalize=False, pre_znormalize=False, predicted_mean_fbest=False,
                    input_znormalize=False, max_dataset_size=10_000, remove_features_with_one_value_only=False,
                    return_actual_ei=False, acq_function='ei', ucb_rest_prob=.05, ensemble_log_dims=False,
                    ensemble_type='mean_probs', # in ('mean_probs', 'max_acq')
                    input_power_transform=False, power_transform_eps=.0, input_power_transform_eps=.0,
                    input_rank_transform=False, ensemble_input_rank_transform=False,
                    ensemble_power_transform=False, ensemble_feature_rotation=False,
                    style=None, outlier_stretching_interval=0.0, verbose=False, unsafe_power_transform=False,
                         ):
    """
    Differences to HEBO:
        - The noise can't be set in the same way, as it depends on the tuning of HPs via VI.
        - Log EI and PI are always used directly instead of using the approximation.

    This is a stochastic function, relying on torch.randn

    :param model:
    :param x_given: torch.Tensor of shape (N, D)
    :param y_given: torch.Tensor of shape (N, 1) or (N,)
    :param x_eval: torch.Tensor of shape (M, D)
    :param kappa:
    :param eps:
    :return:
    """
    assert ensemble_type in ('mean_probs', 'max_acq')
    if rand_sample is not False \
        and (len(x_given) == 0 or
             ((1 + x_given.shape[1] if rand_sample is None else max(2, rand_sample)) > x_given.shape[0])):
        print('rando')
        return torch.zeros_like(x_eval[:,0]) #torch.randperm(x_eval.shape[0])[0]
    y_given = y_given.reshape(-1)
    assert len(y_given) == len(x_given)
    if apply_power_transform:
        if pre_normalize:
            y_normed = y_given / y_given.std()
            if not torch.isinf(y_normed).any() and not torch.isnan(y_normed).any():
                y_given = y_normed
        elif pre_znormalize:
            y_znormed = (y_given - y_given.mean()) / y_given.std()
            if not torch.isinf(y_znormed).any() and not torch.isnan(y_znormed).any():
                y_given = y_znormed
        y_given = general_power_transform(y_given.unsqueeze(1), y_given.unsqueeze(1), power_transform_eps, less_safe=unsafe_power_transform).squeeze(1)
        
        ############################################################################################################
        # Changes for CEI:
        GX = -GX
        GX_t = general_power_transform(GX, GX, power_transform_eps, less_safe=unsafe_power_transform)
        G_thres = general_power_transform(GX, torch.zeros((1, GX.shape[1])).to(GX.device), power_transform_eps, less_safe=unsafe_power_transform)
        GX = GX_t
        ############################################################################################################
        
        
        
        if verbose:
            print(f"{y_given=}")
        #y_given = torch.tensor(power_transform(y_given.cpu().unsqueeze(1), method='yeo-johnson', standardize=znormalize), device=y_given.device, dtype=y_given.dtype,).squeeze(1)
    y_given_std = torch.tensor(1., device=y_given.device, dtype=y_given.dtype)
    if znormalize and not apply_power_transform:
        if len(y_given) > 1:
            y_given_std = y_given.std()
        y_given_mean = y_given.mean()
        y_given = (y_given - y_given_mean) / y_given_std

    if remove_features_with_one_value_only:
        x_all = torch.cat([x_given, x_eval], dim=0)
        only_one_value_feature = torch.tensor([len(torch.unique(x_all[:,i])) for i in range(x_all.shape[1])]) == 1
        x_given = x_given[:,~only_one_value_feature]
        x_eval = x_eval[:,~only_one_value_feature]

    if outlier_stretching_interval > 0.:
        tx = torch.cat([x_given, x_eval], dim=0)
        m = outlier_stretching_interval
        eps = 1e-10
        small_values = (tx < m) & (tx > 0.)
        tx[small_values] = m * (torch.log(tx[small_values] + eps) - math.log(eps)) / (math.log(m + eps) - math.log(eps))

        large_values = (tx > 1. - m) & (tx < 1.)
        tx[large_values] = 1. - m * (torch.log(1 - tx[large_values] + eps) - math.log(eps)) / (
                    math.log(m + eps) - math.log(eps))
        x_given = tx[:len(x_given)]
        x_eval = tx[len(x_given):]

    if input_znormalize: # implementation that relies on the test set, too...
        std = x_given.std(dim=0)
        std[std == 0.] = 1.
        mean = x_given.mean(dim=0)
        x_given = (x_given - mean) / std
        x_eval = (x_eval - mean) / std

    if input_power_transform:
        x_given = general_power_transform(x_given, x_given, input_power_transform_eps)
        x_eval = general_power_transform(x_given, x_eval, input_power_transform_eps)

    if input_rank_transform is True or input_rank_transform == 'full': # uses test set x statistics...
        x_all = torch.cat((x_given,x_eval), dim=0)
        for feature_dim in range(x_all.shape[-1]):
            uniques = torch.sort(torch.unique(x_all[..., feature_dim])).values
            x_eval[...,feature_dim] = torch.searchsorted(uniques,x_eval[..., feature_dim]).float() / (len(uniques)-1)
            x_given[...,feature_dim] = torch.searchsorted(uniques,x_given[..., feature_dim]).float() / (len(uniques)-1)
    elif input_rank_transform is False:
        pass
    elif input_rank_transform == 'train':
        x_given = rank_transform(x_given, x_given)
        x_eval = rank_transform(x_given, x_eval)
    elif input_rank_transform.startswith('train'):
        likelihood = float(input_rank_transform.split('_')[-1])
        if torch.rand(1).item() < likelihood:
            print('rank transform')
            x_given = rank_transform(x_given, x_given)
            x_eval = rank_transform(x_given, x_eval)
    else:
        raise NotImplementedError


    # compute logits
    criterion: bar_distribution.BarDistribution = model.criterion
    x_predict = torch.cat([x_given, x_eval], dim=0)


    logits_list = []
    for x_feed in torch.split(x_predict, max_dataset_size, dim=0):
        x_full_feed = torch.cat([x_given, x_feed], dim=0).unsqueeze(1)
        y_full_feed = y_given.unsqueeze(1)
        if ensemble_log_dims == '01':
            x_full_feed = log01_batch(x_full_feed)
        elif ensemble_log_dims == 'global01' or ensemble_log_dims is True:
            x_full_feed = log01_batch(x_full_feed, input_between_zero_and_one=True)
        elif ensemble_log_dims == '01-10':
            x_full_feed = torch.cat((log01_batch(x_full_feed)[:, :-1], log01_batch(1. - x_full_feed)), 1)
        elif ensemble_log_dims == 'norm':
            x_full_feed = lognormed_batch(x_full_feed, len(x_given))
        elif ensemble_log_dims is not False:
            raise NotImplementedError

        if ensemble_feature_rotation:
            x_full_feed = torch.cat([x_full_feed[:, :, (i+torch.arange(x_full_feed.shape[2])) % x_full_feed.shape[2]] for i in range(x_full_feed.shape[2])], dim=1)

        if ensemble_input_rank_transform == 'train' or ensemble_input_rank_transform is True:
            x_full_feed = torch.cat([rank_transform(x_given, x_full_feed[:,i,:])[:,None] for i in range(x_full_feed.shape[1])] + [x_full_feed], dim=1)

        if ensemble_power_transform:
            assert apply_power_transform is False
            y_full_feed = torch.cat((general_power_transform(y_full_feed, y_full_feed, power_transform_eps), y_full_feed), dim=1)


        if style is not None:
            if callable(style):
                style = style()

            if isinstance(style, torch.Tensor):
                style = style.to(x_full_feed.device)
            else:
                style = torch.tensor(style, device=x_full_feed.device).view(1, 1).repeat(x_full_feed.shape[1], 1)

        ################################################################################
        ################################################################################
        # Changes for CEI
        ################################################################################
        ################################################################################

        logits = model(
            (style,
             x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]+GX.shape[1]),
             torch.cat([y_full_feed, GX], dim=1).unsqueeze(2) ),
            single_eval_pos=len(x_given)
        )
        
        ################################################################################
        ################################################################################


        
        ensemble_type == None
        
    
    logits = logits.softmax(-1).log_()

    
    
    logits_given = logits[:len(x_given)]
    logits_eval = logits[len(x_given):]
    
    objective_given = logits_given[:,0,:].unsqueeze(1)
    objective_eval  = logits_eval[:,0,:].unsqueeze(1)
    constraint_given = logits_given[:,1:,:]
    constraint_eval  = logits_eval[:,1:,:]

    ################################################################################
    
    def acq_ensembling(acq_values): # (points, ensemble dim)
        return acq_values.max(1).values
    
    ################################################################################
    ################################################################################
    # Changes for CEI
    
    # Objective
    tau = torch.max(y_given)
    objective_acq_value = acq_ensembling(criterion.ei(objective_eval, tau))
    
    # Constraints
    constraints_acq_value = acq_ensembling(criterion.pi(constraint_eval[:,0,:].unsqueeze(1), G_thres[0, 0].item()))
    constraints_acq_value = constraints_acq_value.unsqueeze(1)

    
    for jj in range(1,constraint_eval.shape[1]):
        next_constraints_acq_value = acq_ensembling(criterion.pi(constraint_eval[:,jj,:].unsqueeze(1), G_thres[0, jj].item()))
        next_constraints_acq_value = next_constraints_acq_value.unsqueeze(1)
        constraints_acq_value = torch.cat([constraints_acq_value,next_constraints_acq_value], dim=1)

    ################################################################################
    ################################################################################

    return objective_acq_value, constraints_acq_value



class PFNCEI_TransformerBOMethod:

    def __init__(self, model, acq_f=PFNCEI_general_acq_function, device='cpu:0', fit_encoder=None, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder

    @torch.no_grad()
    def observe_and_suggest(self, X_obs, y_obs, X_pen, GX, return_actual_ei=False):
        # assert X_pen is not None
        # assumptions about X_obs and X_pen:
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)
        
        # X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
        # y_obs = to_tensor(y_obs, device=self.device).to(torch.float32).view(-1)
        # X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)
        # GX = to_tensor(GX, device=self.device).to(torch.float32)

        X_obs = X_obs.to(torch.float32)
        y_obs = y_obs.to(torch.float32) #.view(-1)
        X_pen = X_pen.to(torch.float32)
        GX = GX.to(torch.float32)

        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)

        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        # with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
        obj_acq_values, constraint_acq_values = self.acq_function(self.model, X_obs, y_obs,
                                       X_pen, GX, return_actual_ei=return_actual_ei, **self.kwargs) #.cpu().clone()  # bool array
        return obj_acq_values, constraint_acq_values


################################################################################
## End of PFNCEI
################################################################################































