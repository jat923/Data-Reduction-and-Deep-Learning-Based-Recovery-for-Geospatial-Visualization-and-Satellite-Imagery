# Data-Reduction-and-Deep-Learning-Based-Recovery-for-Geospatial-Visualization-and-Satellite-Imagery
## Introduction

Big datasets often demand high-bandwidth, high-capacity storage, and high-performance computational power which makes it difficult for data scientists and decision-makers to analyze and visualize data. A potential approach to mitigate such problems is to reduce big datasets into smaller ones, which will not only lower storage requirements but also allow light load transfer over the network. Carefully prepared data by removing redundancies, along with a machine learning model capable of reconstructing the whole dataset from its reduced version, can improve the storage scalability, data transfer, and speed up the overall data management pipeline.

This repository contains code for data deletion strategies and recovery methods.

## Data Reduction Strategies

* Uniform deletion
   * Grid deletion
   * Checkerboard deletion
* Variance based deletion

## Recovery
* Bayesian Ridge Regression
* Shallow neural network
* SRGAN
* Image inpainting


## Using this repository
### For runing data reduction codes:
* Update the directory name.
* Run 'python batchImageGridDeletion.py'.
### For runing recovery codes:
* Put the prepared training data using the data reduction codes in folder.
* Update the directory in training codes.
* Put the prepared testing data using the data reduction codes in folder.
* Update the directory in testing codes.
### Dependencies
1. OpenCV

2. matplotlib

3. scikit-learn

4. keras

5. tensorflow
