# small file to simply load the cifar datasets into the /data folder

from dataloaders.cifar import CifarDataloader
from pathlib import Path

datapath = Path('./data')
if not datapath.exists():
    datapath.mkdir(exist_ok=True)

noise_rate = 0.0
noise_mode="sym"
batch_size=128
datapath=datapath
noise_file = "NoNoiseCifar10"

# Download cifar10
dataLoader = CifarDataloader(dataset="cifar10", noise_rate=noise_rate, noise_mode=noise_mode, 
                                     batch_size=batch_size, datapath=datapath, noise_file = noise_file, valid_seed = 1)
# Download cifar100
dataLoader = CifarDataloader(dataset="cifar100", noise_rate=noise_rate, noise_mode=noise_mode, 
                                     batch_size=batch_size, datapath=datapath, noise_file = noise_file, valid_seed = 1)
                                     