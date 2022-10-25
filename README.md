# Detecting-Multivariate-Time-Series-Anomalies-with-Zero-Known-Label
This repository provides a PyTorch implementation of MTGFlow, which is the unsupervised anomaly dection and localization method. T 
This repository is based on [`GANF`](https://github.com/EnyanDai/GANF).
## Requirements
* python==3.8.5 
* pytorch==1.7.1
* numpy==1.19.2
* torchvision==1.5
* scipy==1.6.1
* scikit-learn==0.24.1
* scikit-image==0.18.1
* matplotlib== 3.3.4
* pillow == 7.2.0


```sh
pip install -r requirements.txt
```

## Data
We test our method for five public datasets, e.g., SWaT, WADI, PSM, MSL, and SMD.

## Train

For example, training for WADI
```sh
sh runners/run_WADI.sh
```
## Test
We provide the pretained model.

For example, testing for WADI 
```sh
sh runners/run_WADI_test.sh
```
