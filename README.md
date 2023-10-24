# TGAE: Temporal Graph Autoencoder for Travel Forecasting

This is the PyTorch implementation of the paper:

Q. Wang, H. Jiang, M. Qiu, Y. Liu and D. Ye, "TGAE: Temporal Graph Autoencoder for Travel Forecasting," in IEEE Transactions on Intelligent Transportation Systems, 2023, doi: 10.1109/TITS.2022.3202089.

## Requirements
- python==3.8.8
- networkx
- numpy
- pandas
- sklearn
- torch==1.9.0
- torch-cluster==1.5.9
- torch-scatter==2.0.9
- torch-sparse==0.6.12
- torch-spline-conv==1.2.1
- torchvision==0.10.0
- torch-geometric==2.0.4

## Data
The pre-processed data is under the folder `data`.

## Run
1. Specify the arguments in the `train.py`.
2. Run the code by `python train.py`.

## Citation
Please cite the following paper, if you find the repository or the paper useful.

@ARTICLE{9889163,  
author={Wang, Qiang and Jiang, Hao and Qiu, Meikang and Liu, Yifeng and Ye, Dongsheng},  
journal={IEEE Transactions on Intelligent Transportation Systems},   
title={TGAE: Temporal Graph Autoencoder for Travel Forecasting},   
year={2023},
volume={24},
number={8},
pages={8529-8541}, 
doi={10.1109/TITS.2022.3202089}}
