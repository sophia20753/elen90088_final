'''
Script Name: test_model_tour.py

Description:
    This script runs a simulation in the Duckietown environment using a pre-trained model (either DNN or SVR) and collects data during the simulation. The script uses the model to predict steering angles, and saves the collected data to csv files for further analysis and evaluation.

Usage Instructions:
    To run this script, use the following command:
        python test_model_tour.py --model_name <model_name>
    Example:
        python test_model_tour.py --model_name base_dnn_raw_pd

Parameters:
    --model_name: Name of the model file to use for predictions (str). The name should follow the format:
        <version>_<type>_<unused>_<data>_<controller>. Example: base_dnn_model_raw_pd.

Original Code Reference:
    This script was adapted from code provided by Professor Tansu Alpcan for ELEN90088.

Author:
    Sophia Chiang
    June 12th 2024
'''
# Add path to models
import sys
import os

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to the Python path
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.append(src_dir)

# Import relevant libraries
import pickle
from pickle import FALSE, TRUE
from PIL import Image
import gym
import numpy as np
import pandas as pd
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from project_functions import save_data_to_csv

def process_input(env, reward, mean, scale, model_version):
    '''Function to prepare input data for model prediction'''
    lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_position.dist
    angle_from_straight_in_rads = lane_position.angle_rad

    # Normalise observation data 
    distance_to_road_center = (distance_to_road_center-mean[1])/scale[1]
    angle_from_straight_in_rads = (angle_from_straight_in_rads-mean[2])/scale[2]

    if model_version == 'unweighted':
        # Input data for unweighted svr
        input_data = np.array([[distance_to_road_center, angle_from_straight_in_rads, reward]])
    else:
        # Input data for dnn & weighted svr
        input_data = np.array([[distance_to_road_center, angle_from_straight_in_rads]])

    return input_data

def main(model_name):
    # Parse model_name
    string = model_name.split('_')
    model_version = string[0] # base/reg/best or unweighted/weighted
    model_type = string[1] # svr/dnn
    data_type = string[3] # raw/processed
    controller = string[4] # pd/mpc

    map_name = 'udem1' # Test map

    # Create environment with testing map
    env = DuckietownEnv(map_name=map_name, domain_rand=False, draw_bbox=False, user_tile_start=(1, 1))
    env.render()
    env.max_steps = 5000 # Increase max step count to ensure robot finishes traversing map

    if model_type == 'dnn':
        # Load model for dnn
        model = load_model(f"models/{model_name}.h5")
    else:
        # Load model for svr
        with open(f'models/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)

    # Load scaling parameters
    with open(f'train_data/{data_type}_data_{controller}_scaling_params.pkl', 'rb') as file:
        scaling_params = pickle.load(file)
    mean = scaling_params['mean']
    scale = scaling_params['scale']

    # Create folder for test data
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
    
    # Initialise test data dict
    test_model_data = {
        "cur_pos_x": [],
        "cur_pos_z": [],
        "distance_to_road_center": [],
        "angle_from_straight_in_rads": [],
        "steering_angle": [],
        "reward": [],
    }

    # Make initial step with default speed and angle
    car_speed = 0.1
    steering_angle = 0

    _, reward, done, _ = env.step([car_speed, steering_angle])

    # Adapted from Professor Alpcan's example code
    while True:
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        input_data = process_input(env, reward, mean, scale, model_version)

        # Make prediction
        steering_angle = model.predict(input_data)

        # Reverse normalisation
        steering_angle = steering_angle*scale[0] + mean[0]

        _, reward, done, _ = env.step([car_speed, steering_angle])

        # Update display
        env.render()

        # Store timestep data
        test_model_data["cur_pos_x"].append(env.cur_pos[0])
        test_model_data["cur_pos_z"].append(env.cur_pos[2])
        test_model_data["distance_to_road_center"].append(distance_to_road_center)
        test_model_data["angle_from_straight_in_rads"].append(angle_from_straight_in_rads)
        test_model_data["steering_angle"].append(steering_angle)
        test_model_data["reward"].append(reward)

        if done:
            if reward < 0:
                print(f"Test crashed, run again")
                env.reset()
                # Reset data collection 
                test_model_data = {
                    "cur_pos_x": [],
                    "cur_pos_z": [],
                    "distance_to_road_center": [],
                    "angle_from_straight_in_rads": [],
                    "steering_angle": [],
                    "reward": [],
                }
                continue
            else:
                # Successful run so save data to csv
                print(f"Test reached goal")
                save_data_to_csv("test_data", f"test_{model_name}_data.csv", test_model_data)
                break

    

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Run Duckietown simulation and collect data')
    parser.add_argument('--model_name', type=str, help='Name of model file')
    args = parser.parse_args()

    main(args.model_name)