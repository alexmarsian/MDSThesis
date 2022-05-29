from peer.peer_core import *
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.resnet import *
from tools.prune import *
import numpy as np
import time

def train(epoch, model, optimizer, data_loader, device, 
          loss_criterion=F.cross_entropy, mask=None,
         peer_loader = None):
    """
    Trains the model for one epoch
    """
    train_loss = 0.
    train_acc = 0
    N = 0
    
    model.train()    
    for batch_x, batch_y in data_loader:
        # load batch onto device
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        # get outputs and calculate loss
        outputs = model(batch_x)
        if peer_loader is not None:
            # Prepare mixmatched images and labels for the Peer Term
            peer_iter = iter(peer_loader)
            input1 = peer_iter.next()[0]
            output1 = model(input1.to(device))
            target2 = peer_iter.next()[1]
            target2 = torch.Tensor(target2.float())
            target2 = torch.autograd.Variable(target2.to(device))
            # Peer Loss with Cross-Entropy loss: L(f(x), y) - L(f(x1), y2)
            loss = loss_criterion(outputs, batch_y.long()) - f_alpha(epoch) * loss_criterion(output1, target2.long())
            loss.to(device)
            loss.backward()
        else:
            loss = loss_criterion(outputs, batch_y)
            # backpropagate and update optimizer
            loss.backward()
        if mask is not None: mask.step() # required for Sparse Evolutionary Training
        else: optimizer.step()
        train_loss += loss.item() * len(batch_x)
        # calculate training accuracy
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        N += len(batch_x)
    
    train_loss /= N
    train_acc /= N
    return train_loss, train_acc

def evaluate(model, data_loader, device, loss_criterion=F.cross_entropy):
    """
    Evaluates the model for one epoch
    """
    val_loss = 0.
    val_acc = 0
    N = 0
    
    model.eval()  
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = loss_criterion(outputs, batch_y)
            val_loss += loss.item() * len(batch_x)
            pred = torch.max(outputs, 1)[1]
            val_correct = (pred == batch_y).sum()
            val_acc += val_correct.item()
            N += len(batch_x)
    
    val_loss /= N
    val_acc /= N
    return val_loss, val_acc

def get_model(dataset, lr=0.1, sparsity = 0.0):
    """
    Returns the appropriate model
    """
    if dataset == "cifar10":
        model = ResNet18(10)
    else:
        model = ResNet34(100)
    # apply pruning
    if sparsity > 0:
        prune_weights_reparam(model)
        prune_weights_erk(model, sparsity)
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # settings as in the Resnet paper
        
    return model, optimizer