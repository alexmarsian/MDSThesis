import torch.nn as nn
import torch
import numpy as np

class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )
    
    
def f_alpha(epoch):
    # Uniform/Random noise setting
    alpha1 = np.linspace(0.0, 0.0, num=20)
    alpha2 = np.linspace(0.0, 0.1, num=20)
    alpha3 = np.linspace(1, 2, num=50)
    alpha4 = np.linspace(2, 2.5, num=50)
    alpha5 = np.linspace(2.5, 3.3, num=100)
    alpha6 = np.linspace(3.3, 5, num=100)
     
    alpha = np.concatenate((alpha1, alpha2, alpha3, alpha4, alpha5, alpha6),axis=0)
    return alpha[epoch]
