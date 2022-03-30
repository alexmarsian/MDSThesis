# Preliminaries. Not to be exported.
import torch
import torch.nn as nn

def _is_prunable_module(m):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d))

def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz/total
    
def _get_nnz(tsr):
    return tsr.nonzero().size(0)

# Modules

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights

def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules