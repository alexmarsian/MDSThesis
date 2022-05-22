import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import torch

def unpickle(file):
    """
    Helper function to load the pickle files
    """
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def split(images, labels, valid_percentage, seed):
    """
    A helper function to create a validation split.
    """
    train_split = 1-valid_percentage
    N = len(labels)
    index = np.arange(N)
    if seed is not None:
        r = np.random.RandomState(seed)
        r.shuffle(index)
    else:
        np.random.shuffle(index)
    
    train_idx, val_idx = index[:int(train_split * N)], index[int(train_split * N):]
    labels = np.array(labels)

    train_images, train_labels = images[train_idx], labels[train_idx]
    val_images, val_labels = images[val_idx], labels[val_idx]
    
    return train_images, list(train_labels), val_images, list(val_labels)
    

# make Dataset Class where noise can be added
class CifarDataset(Dataset):
    def __init__(self, dataset, noise_rate, noise_mode, datapath, 
                 transform, noise_file, train=True, valid_pct = 0.2, valid=False, seed = 1):
            
        # define class variables
        self.nr = noise_rate # noise rate
        self.transform = transform
        if dataset=='cifar10':
            self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise (same as divideMix for CIFAR10)
        elif dataset=='cifar100':
            self.transition = {c:np.random.randint(100) for c in range(100)} # class transition for asymmetric noise (random)
        self.train = train
        
        # load test data
        if not train:
            if dataset=='cifar10':
                dp = datapath / Path('cifar-10-batches-py')
                test_dic = unpickle(f'{dp}/test_batch')
                self.data = test_dic['data']
                self.data = self.data.reshape((10000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))  
                self.labels = test_dic['labels']
            elif dataset=='cifar100':
                dp = datapath / Path('cifar-100-python')
                test_dic = unpickle(f'{dp}/test')
                self.data = test_dic['data']
                self.data = self.data.reshape((10000, 3, 32, 32))
                self.data = self.data.transpose((0, 2, 3, 1))  
                self.labels = test_dic['fine_labels']
        # load train data
        else:
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                dp = datapath / Path('cifar-10-batches-py')
                for n in range(1,6):
                    dpath = f'{dp}/data_batch_{n}'
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':  
                dp = datapath / Path('cifar-100-python')
                train_dic = unpickle(f'{dp}/train')
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            # create noisy labels
            if noise_mode == 'human':
                if dataset == 'cifar10':
                    noise_file = torch.load('./data/CIFAR-10_human.pt')
                    noise_label = noise_file['random_label1']
                elif dataset == 'cifar100':
                    noise_file = torch.load('./data/CIFAR-100_human.pt')
                    noise_label = noise_file['noisy_label']
            else:
                noise_file = datapath / Path('noisyLabels') / Path(noise_file)
                if noise_file.exists():
                    noise_label = json.load(open(noise_file,"r"))
                else:    #inject noise   
                    noise_label = []
                    idx = list(range(50000))
                    random.shuffle(idx)
                    num_noise = int(self.nr*50000)            
                    noise_idx = idx[:num_noise]
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)
                            elif noise_mode=='asym':   
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)                    
                        else:    
                            noise_label.append(train_label[i])
                    # save noisy labels file to allow repeated experiments
                    print(f"saving noisy labels to {noise_file} ...")
                    noise_file.parent.absolute().mkdir(exist_ok=True)
                    with open(noise_file, 'w') as f:
                        json.dump(noise_label,f)
                
            train_data, train_labels, val_data, val_labels = split(train_data, noise_label, valid_pct, seed)
            # if you don't want a validation split, set valid_pct to 0.0 and valid = False, otherwise split is made by default
            if valid:
                self.data, self.labels = val_data, val_labels
            else:
                self.data, self.labels = train_data, train_labels
            
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        img = self.transform(img)            
        return img, target
           
    def __len__(self):
        return len(self.data)
        

class CifarDataloader():  
    def __init__(self, dataset, noise_rate, noise_mode, batch_size, datapath, noise_file='noisylabels.json', 
    num_workers=2, valid_seed = 1, valid_pct=0.2):
        # set transforms for training and test data
        if dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])

        train_data = CifarDataset(dataset, noise_rate, noise_mode, datapath=datapath, 
                                  transform=self.transform_train, noise_file=noise_file, valid = False, seed=valid_seed, valid_pct=valid_pct)                
        train_loader = DataLoader(
            dataset=train_data, 
            batch_size=batch_size*2,
            shuffle=True,
            num_workers=num_workers)
        self.trainLoader = train_loader
        
        valid_data = CifarDataset(dataset, noise_rate, noise_mode, datapath=datapath, 
                                  transform=self.transform_test, noise_file=noise_file, valid = True, seed=valid_seed, valid_pct=valid_pct)                
        valid_loader = DataLoader(
            dataset=valid_data, 
            batch_size=batch_size*2,
            shuffle=False,
            num_workers=num_workers)
        self.validLoader = valid_loader

        test_dataset = CifarDataset(dataset, noise_mode, noise_rate, datapath=datapath, transform=self.transform_test, noise_file=noise_file, train=False)      
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)          
        self.testLoader = test_loader