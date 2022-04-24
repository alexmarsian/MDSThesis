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
        datapath, noise_file, weightFileName, repeats):
    """
    A wrapper function to run the training loop (train, evaluate, test) with specified parameters.
    """
    # intialise device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # same optimisation and epoch schedule used
    epochs = 160
    
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

        # initialise model and learning rate schedule
        model, optimizer = get_model(dataset, lr=0.1, sparsity=sparsity)
        schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1) # schedule based on Resnet paper ("Deep residual networks for...")
        model = model.to(device)

        # save the model for best validaiton loss throughout training    
        best_epoch, best_val_loss = 0, np.inf
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []

        for epoch in range(epochs):

            start = time.time()
            print("Epoch: {}".format(epoch+1))
            f.write("Epoch: {}\n".format(epoch+1))

            train_loss, train_acc = train(model, optimizer, train_loader, device)
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
                torch.save(model.state_dict(), weightsDir / Path(f'best_forward_exp3.pkl'))

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
        model.load_state_dict(torch.load(weightsDir / Path(f'best_forward_exp3.pkl'), map_location=device))
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
        
# Dense, 50% Symmetric Noise, Cifar100
dataset = "cifar100"
noise_rate = 0.5
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "50SymNoiseCifar100"
sparsity = 0
weightFileName = f"R34_Cifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)
 
# Dense, 80% Symmetric Noise, Cifar100
dataset = "cifar100"
noise_rate = 0.8
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "80SymNoiseCifar100"
sparsity = 0
weightFileName = f"R34_Cifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)
 
# Dense, 90% Symmetric Noise, Cifar100
dataset = "cifar100"
noise_rate = 0.9
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "90SymNoiseCifar100"
sparsity = 0
weightFileName = f"R34_Cifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)

# Sparse, 50% symmetric noise, Cifar100
dataset = "cifar100"
noise_rate = 0.5
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "50SymNoiseCifar100"
sparsity = 0.8
weightFileName = f"R34_20SparseCifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)


# Sparse, 80% symmetric noise, Cifar100
dataset = "cifar100"
noise_rate = 0.8
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "80SymNoiseCifar100"
sparsity = 0.8
weightFileName = f"R34_20SparseCifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)


# Sparse, 90% symmetric noise, Cifar100
dataset = "cifar100"
noise_rate = 0.9
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "90SymNoiseCifar100"
sparsity = 0.8
weightFileName = f"R34_20SparseCifar100_{int(noise_rate*100)}pct_{noise_mode}"
repeats = 3

# function to run training and evaluation loop
run(dataset, noise_rate, noise_mode, sparsity, batch_size, 
    datapath, noise_file, weightFileName, repeats)

