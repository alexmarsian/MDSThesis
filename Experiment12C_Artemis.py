import time
from itop.core import Masking, CosineDecay, LinearDecay
from peer.peer_core import *
import torch.nn.functional as F
from dataloaders.cifar_artemis import CifarDataloader
from train import *
from pathlib import Path
import numpy as np

# Directory for downloading and reading training data from
datapath = Path('./data')
if not datapath.exists():
    datapath.mkdir(exist_ok=True)
# Directory for saving model weights during training
weightsDir = Path('models/weights')
weightsDir.mkdir(exist_ok=True)
# Directory for saving results during training
resultsDir = Path('results')
resultsDir.mkdir(exist_ok=True)

def run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
        datapath, noise_file, weightFileName, repeats, sparse_args={}, peer=False):
    """
    A wrapper function to run the training loop (train, evaluate, test) with specified parameters.
    """
    # intialise device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Epochs set at 200 for all non-SET experiments
    epochs = 200 if sparse_args == {} else 200*sparse_args['multiplier']
    
    # running averages
    avgTrainAcc, avgTrainLoss = [],[]
    avgValAcc, avgValLoss = [],[]
    avgTestAcc, avgTestLoss = [],[]
    
    
    for r in range(repeats):
        f = open(resultsDir / Path(weightFileName + f'_{r}.txt'), 'w') # save results for this run

        # get data loaders for training, validation, and test sets
        # different seed used for every repeat to split the training into train + valid set
        dataLoader = CifarDataloader(dataset=dataset, noise_rate=noise_rate, noise_mode=noise_mode, 
                                     batch_size=batch_size, datapath=datapath, noise_file = noise_file, valid_seed = r)
        train_loader = dataLoader.trainLoader
        val_loader = dataLoader.validLoader
        test_loader = dataLoader.testLoader
        n = len(train_loader.dataset)
        
        # Get dataloader for peer loss if required
        if peer:
            dataLoader = CifarDataloader(dataset=dataset, noise_rate=noise_rate, noise_mode=noise_mode, 
                                     batch_size=batch_size, datapath=datapath, noise_file = noise_file, valid_seed = r)
            peer_train = dataLoader.trainLoader
            criterion = CrossEntropyLossStable()
            criterion.to(device)
            

        # initialise model and learning rate schedule
        model, optimizer = get_model(dataset, lr=0.1, sparsity=sparsity)
        # Set schedule for SET/RigL (based on In-Time-Over-Parameterisation paper)
        if sparse_args != {}:
            schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs / 2) * sparse_args['multiplier'], int(epochs * 3 / 4) * sparse_args['multiplier']], last_epoch=-1)
        # Set schedule for non-SET models
        else:
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1) # schedule based on Resnet paper
        model = model.to(device)
        
        # Initialise mask (remains None for non-set models)
        mask = None
        if sparse_args != {}:
            decay = CosineDecay(sparse_args['death_rate'], len(train_loader)*(epochs*sparse_args['multiplier']))
            mask = Masking(optimizer, death_rate=sparse_args['death_rate'], death_mode=sparse_args['death_mode'], death_rate_decay=decay, growth_mode=sparse_args['growth_mode'],
                           redistribution_mode=sparse_args['redistribution'], args=sparse_args) 
            mask.add_module(model, sparse_init=sparse_args['sparse_init'], density=1-sparsity)

        # save the model for best validaiton loss throughout training    
        best_epoch, best_val_loss = 0, np.inf
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []

        for epoch in range(epochs):

            start = time.time()
            print("Epoch: {}".format(epoch+1))
            f.write("Epoch: {}\n".format(epoch+1))
            
            if peer:
                train_loss, train_acc = train(epoch, model, optimizer, train_loader, device, mask=mask, peer_loader=peer_train, loss_criterion = criterion)
            else:
                train_loss, train_acc = train(epoch, model, optimizer, train_loader, device, mask=mask)
            train_loss /= n
            print("Train Loss: {:.6f}, Acc: {:.6f}".format(train_loss, train_acc))
            f.write("Train Loss: {:.6f}, Acc: {:.6f}\n".format(train_loss, train_acc))

            val_loss, val_acc = evaluate(model, val_loader, device)
            print("Val Loss: {:.6f},  Acc: {:.6f}".format(val_loss, val_acc))
            f.write("Val Loss: {:.6f},  Acc: {:.6f}\n".format(val_loss, val_acc))

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_train_loss = train_loss
                torch.save(model.state_dict(), weightsDir / Path(f'{weightFileName}_best_forward_.pkl'))

            schedule.step()

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            end = time.time()
            print("Took {:.2f} minutes.\n".format((end - start) / 60))
            f.write("Took {:.2f} minutes.\n".format((end - start) / 60))

        print("Best epoch:", best_epoch+1)
        f.write(f"Best epoch: {best_epoch+1}\n")
        print("Final train loss: {:.4f}".format(train_loss_list[best_epoch]))
        f.write("Final train loss: {:.4f}\n".format(train_loss_list[best_epoch]))
        print("Final train accuracy: {:.4f}\n".format(train_acc_list[best_epoch]))
        f.write("Final train accuracy: {:.4f}\n".format(train_acc_list[best_epoch]))
        print("Final val loss: {:.4f}".format(val_loss_list[best_epoch]))
        f.write("Final val loss: {:.4f}\n".format(val_loss_list[best_epoch]))
        print("Final val accuracy: {:.4f}\n".format(val_acc_list[best_epoch]))
        f.write("Final val accuracy: {:.4f}\n".format(val_acc_list[best_epoch]))

        # load best model saved during training
        model.load_state_dict(torch.load(weightsDir / Path(f'{weightFileName}_best_forward_.pkl'), map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, device)
        print("Test accuracy: {:.4f}".format(test_acc))
        f.write("Test accuracy: {:.4f}\n".format(test_acc))
        
        # save average results
        avgTrainAcc.append(train_acc_list[best_epoch])
        avgTrainLoss.append(train_loss_list[best_epoch])
        avgValAcc.append(val_acc_list[best_epoch])
        avgValLoss.append(val_loss_list[best_epoch])
        avgTestAcc.append(test_acc)
        avgTestLoss.append(test_loss)
                                                           
        # Write final Mask update to file if SET/RigL
        if sparse_args != {}:
            layer_fired_weights, total_fired_weights = mask.fired_masks_update()
            for name in layer_fired_weights:
                f.write(f'The final percentage of fired weights in the layer {name} is: {layer_fired_weights[name]}')
            f.write(f'The final percentage of the total fired weights is: {total_fired_weights}')

        f.close()
        
    # save average results for this run    
    with open(resultsDir / Path(weightFileName + f'_avg.txt'), 'w') as f:
        
        f.write(f"Repeats: {repeats}\n")
        f.write(f"Average Training Accuracy: {np.mean(avgTrainAcc)}\n")
        f.write(f"Stdev Training Accuracy: {np.std(avgTrainAcc)}\n")
        f.write(f"Average Training Loss: {np.mean(avgTrainLoss)}\n")
        f.write(f"Stdev Training Loss: {np.std(avgTrainLoss)}\n")
        f.write(f"Average Validation Accuracy: {np.mean(avgValAcc)}\n")
        f.write(f"Stdev Validation Accuracy: {np.std(avgValAcc)}\n")
        f.write(f"Average Validation Loss: {np.mean(avgValLoss)}\n")
        f.write(f"Stdev Validation Loss: {np.std(avgValLoss)}\n")
        f.write(f"Average Test Accuracy: {np.mean(avgTestAcc)}\n")
        f.write(f"Stdev Test Accuracy: {np.std(avgTestAcc)}\n")
        f.write(f"Average Test Loss: {np.mean(avgTestLoss)}\n")
        f.write(f"Stdev Test Loss: {np.std(avgTestLoss)}\n")

# Training settings

# Sparse, Cifar10, No Noise, Peer Loss
dataset = "cifar10"
noise_rate = 0.0
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "NoNoiseCifar10"
sparsity = 0.8
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 20% symmetric noise, Peer Loss
noise_rate = 0.2
noise_file = "20SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 30% symmetric noise, Peer Loss
noise_rate = 0.3
noise_file = "30SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 40% symmetric noise, Peer Loss
noise_rate = 0.4
noise_file = "40SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 50% symmetric noise, Peer Loss
noise_rate = 0.5
noise_file = "50SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 80% symmetric noise, Peer Loss
noise_rate = 0.8
noise_file = "80SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 90% symmetric noise, Peer Loss
noise_rate = 0.9
noise_file = "90SymNoiseCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, 40% asymmetric noise, Peer Loss
noise_rate = 0.4
noise_mode="asym"
noise_file = "asymCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

# Sparse, Cifar10, human noise, Peer Loss
noise_rate = 0.0
noise_mode="human"
noise_file = "humanCifar10"
weightFileName = f"R18_Cifar10_peerSparse_{int(noise_rate*100)}pct_{noise_mode}"

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats, peer=True)

