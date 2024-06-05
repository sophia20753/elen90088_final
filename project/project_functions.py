# project_functions.py

# import relevant libraries
import os
import pandas as pd
import numpy as np

def save_data_to_csv(foldername, filename, data):
    '''Saves data from dict to a .csv file'''
    df = pd.DataFrame(data)
    csv_file_path = os.path.join(foldername, filename)
    df.to_csv(csv_file_path, index=False)
    print(f"Saved {filename}")

def huber_loss(ytrue, ypred, delta=1.2):
    """Calculate the Huber loss between true and predicted values"""
    error = ytrue - ypred
    abs_error = np.abs(error)
    
    # Calculate Huber loss
    huber_loss = np.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5 * delta))
    
    # Return the mean Huber loss
    return np.mean(huber_loss)

def huber_loss_coordinates(xtrue, ytrue, xpred, ypred, delta=1.2):
    """Calculate the Huber loss between true and predicted coordinates"""
    error_x = xtrue - xpred
    error_y = ytrue - ypred
    squared_error = (error_x ** 2) + (error_y ** 2)
    huber_loss = np.where(squared_error < (delta ** 2), 0.5 * squared_error, delta * np.sqrt(squared_error) - 0.5 * delta ** 2)
    return np.mean(huber_loss)