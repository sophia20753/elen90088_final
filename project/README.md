# Algorithms and Models for Lane-Keep Assist Systems

This project focuses on developing autonomous vehicle system capable of lane adherance. To do this, we compare the performance of a model-predictive control (MPC) optimisation algorithm, a support vector regression model and a deep neural network, both trained with PD control and MPC control datasets. 

## Usage Overview

### MPC

The control_MPC.py script has the best-fitted, nominated cost formulation for optimal zero-error reference tracking.

To change what environment the script will force itself to run in, add 'map_name = '<insert a map name from the selection above the Simulation object>''.

By same means a starting tile can be chosen by adding, 'start_tile = <[i,j]>' - an array consisting of 2 positive integers

The file will also print the position and heading angle error at every time-step in the command window.

As a further visual aid, the '''--draw-curve''' string can be added at the end of the script call in the command window to produce the optimal paths as coloured lines.

## Project Structure

**MPC scripts**

- `control_MPC.py`: Script to implement MPC and write data to a csv.

**DNN scripts**

In the `project` directory,
- `gen_train.py`: Script to generate training data using PD control.
- `test_pd_tour.py`: Script to generate PD control data in test map.
- `test_model_tour.py`: Script to generate DNN model data in test map.
- `data_preprocess.ipynb`: Notebook to process training data for model training.
- `svr_train.ipynb`: Notebook to train SVR models.
- `dnn_train.ipynb`: Notebook to train DNN models.
- `svr_analysis.ipynb`: Notebook to analyse SVR model performance.
- `dnn_analysis.ipynb`: Notebook to analyse DNN model performance.
- `project_functions.py`: Script that contains module functions used in this project. 
