# TGAE: Temporal Graph Autoencoder for Travel Forecasting
This is the PyTorch implementation of the paper:

Qiang Wang, Hao Jiang, Meikang Qiu, et al., TGAE: Temporal Graph Autoencoder for Travel Forecasting, xxx

## Requirements
- python>=3.6
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
The pre-processed data can be downloaded from [here]. 
Please put the pre-processed data under the folder `data`.

## Run
1. Download the pre-processed data from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0)
   and put it to the folder `data`.
2. Specify the arguments in the `main.py`.
3. Run the code by `python main.py`.

## Citation
Please cite the following paper, if you find the repository or the paper useful.

Qiang Wang, Hao Jiang, Meikang Qiu, et al., TGAE: Temporal Graph Autoencoder for Travel Forecasting, xxx
