# PFN-CEI: A Pre-Trained Transformer-Based Constrained Bayesian Optimization Algorithm


Paper: "Fast and Accurate Bayesian Optimization with Pre-trained Transformers for Constrained Engineering Problems"

## Introduction
PFN-CEI is constrained Bayesian Optimization(CBO) framework using prior-data fitted network (PFN) that can do faster Bayesian Optimization (BO) than using Gaussian process. This framework is built upon [BoTorch](https://github.com/pytorch/botorch) and [PFNs4BO](https://github.com/automl/PFNs4BO). The main contribution is to add the constrain-handling ability to PFNs4BO framework by using the batch processing ability of PFN's transformer structure.  

## CBO algorithms
The tutorials show you how to use three constraint handling methods on PFN-based and GP-based CBO, in total six algorithms. Here are the tutorial of six algorithms:
1. Tutorial_GP_Pen.ipynb:
2. Tutorial_GP_CEI.ipynb:
3. Tutorial_GP_CEI_plus.ipynb:
4. Tutorial_PFN_Pen.ipynb:
5. Tutorial_PFN_CEI.ipynb:
6. Tutorial_PFN_CEI_plus.ipynb: 

![Visual]([http://url/to/img.png](https://github.com/rosenyu304/BO-PFN-CEI/blob/main/image.png?raw=true))

## Test problems
We provided 15 test optimization problems for benchmarking BO methods. The way of using it is shown in Test_function_example.ipynb and here:
```
import torch
import numpy as np

# Select your test case
from test_functions.Ackley2D import Ackley2D, Ackley2D_Scaling

# Initialized sample in the correct dimension based on the test case
# The test case need to have X in [0,1]
X = torch.rand(20,2)

# Scale the X in [0,1] to the domain of interest
X_scaled = Ackley2D_Scaling(X)

# The test case output the gx (constaint) and fx (objective)
gx, fx = Ackley2D(X_scaled)
```
