# Algorithms and Models for Lane-Keep Assist Systems

This project focuses on developing autonomous vehicle system capable of staying within its lanes. To do this, we compare the performance of a model-predictive control (MPC) optimisation algorithm with and without classical machine learning methods, a support vector regression model and a deep neural network trained with PD control and MPC control. 

## MPC Usage

The control_MPC.py script has the best-fitted, nominated cost formulation for optimal zero-error reference tracking.

To change what environment the script will force itself to run in, add 'map_name = '<insert a map name from the selection above the Simulation object>''.

By same means a starting tile can be chosen by adding, 'start_tile = <[i,j]>' - an array consisting of 2 positive integers

The file will also print the position and heading angle error at every time-step in the command window.

As a further visual aid, the '''--draw-curve''' string can be added at the end of the script call in the command window to produce the optimal paths as coloured lines.

## DNN Usage

1. Navigate to directory `project`.

2. Generate training data using specified `num_simulations` to run and `train_map_name` to train in:

    ```python gen_train.py --num_simulations=<num_simulations> --map_name=<train_map_name>```

    Default number of simulations is `10`.
    
    This will save `csv` files for each simulation run in a folder named `train_data_map_name` ready for processing. 

3. Process training data for training using Python notebook `data_preprocess.ipynb` in Jupyter Notebook or JupyterLab (follow instructions in notebook). 

4. Train DNN model using Python notebook `dnn.ipynb` in Jupyter Notebook or JupyterLab (follow instructions in notebook).

5. Run model in Duckietown using specified `test_folder_name` for test data and `map_name` to test in:

    ```python test_dnn_tour.py --folder_name=<test_folder_name> --map_name=<test_map_name>```

6. Run simulation in the same `map_name` using PD control for comparison:

    ```python test_pd_tour.py --folder_name=<test_folder_name> --map_name=<test_map_name>```

7. Analyse results using Python notebook `dnn_analysis.ipynb` in Jupyter Notebook or JupyterLab (follow instructions in notebook).

## Project Structure

**MPC scripts**

- `control_MPC.py`: Script to implement MPC and write data to a csv.

**DNN scripts**

- `gen_train.py`: Script to generate training data.
- `test_pd_tour.py`: Script to generate PD control testing data.
- `test_dnn_tour.py`: Script to generate DNN model testing data.
- `data_preprocess.ipynb`: Notebook to process training data for model training.
- `dnn.ipynb`: Notebook to train DNN model. 
- `dnn_analysis`: Notebook to analyse DNN model performance.
