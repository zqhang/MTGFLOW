# Detecting-Multivariate-Time-Series-Anomalies-with-Zero-Known-Label
This repository provides a PyTorch implementation of MTGFlow, which is the unsupervised anomaly dection and localization method.

## Requirement
* python==3.8.5 
* pytorch==1.7.1
* numpy==1.19.2
* torchvision==1.5
* scipy==1.6.1
* scikit-learn==0.24.1
* scikit-image==0.18.1
* matplotlib== 3.3.4
* pillow == 7.2.0

## Data
We test our method for five public datasets, e.g., SWaT, WADI, PSM, MSL, and SMD.

## Train

Train for WADI
*sh runners/run_WADI.sh

## Test
We provide the pretained model.
Test for WADI 
*sh runners/run_WADI_test.sh
