# MDSThesis

Author: Alexander Mars
Supervised by Vinoth Nandakumar and Tonglian Liu (All from University Of Sydney). 

This code is part of the submission for the Master Of Data Science Capstone project. 
The thesis is titled "Want to get robust? Time to lose weights.", this project focuses on using sparsity for learning with noisy labels. 

To run the code, extract an experiment file from the experiments folder into the parent directory (this directory) and then simply run python3 ExperimentX.py. Scripts for submission to a pbs-based linux computing cluster are also included. Refer to these scripts for the estimated run-time. 

Parts of the code have been modified from these sources:
- Shiwei Liu, Lu Yin, Decebal Constantin Mocanu and Mykola Pechenizkiy, "Do we actually need dense over-parameterization? in-time over-parameterization in sparse training", International Conference on Machine Learning, 2021.
- Multi-class peer loss functions - https://github.com/weijiaheng/Multi-class-Peer-Loss-functions/tree/main/CIFAR-10
- Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, Jinwoo Shin, "Layerwise Sparsity for Magnitude-based Pruning", ICLR 2021.

Please reference this repository if you use it!