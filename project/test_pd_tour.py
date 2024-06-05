# import relevant libraries
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pandas as pd
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.envs.duckietown_env import DuckietownNav
import csv
import random
import argparse
import os
from project_functions import save_data_to_csv

def main(folder_name, map_name):
    # create environment with testing map
    env = DuckietownEnv(map_name=map_name, domain_rand=False, draw_bbox=False, user_tile_start=(1, 1))
    env.render()
    env.max_steps = 5000 # increase max step count to ensure robot finishes traversing map

    # create folder for simulation data
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # initialise test data for analysis
    test_pd_data = {
        "cur_pos_x": [],
        "cur_pos_z": [],
        "distance_to_road_center": [],
        "angle_from_straight_in_rads": [],
        "reward": [],
        "steering_angle": []
    }

    bez_points = {
        'bez_x': [],
        'bez_z': []
    }

    # initialise total reward
    total_reward = 0

    while True:
        lane_position = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        distance_to_road_center = lane_position.dist
        angle_from_straight_in_rads = lane_position.angle_rad

        k_p = 10
        k_d = 1

        car_speed = 0.1 # fixed speed

        # calculate steering angle using PD control
        steering_angle = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads

        obs, reward, done, info = env.step([car_speed, steering_angle])

        # update total reward
        total_reward += reward

        # store timestep data
        test_pd_data["cur_pos_x"].append(env.cur_pos[0])
        test_pd_data["cur_pos_z"].append(env.cur_pos[2])
        test_pd_data["distance_to_road_center"].append(distance_to_road_center)
        test_pd_data["angle_from_straight_in_rads"].append(angle_from_straight_in_rads)
        test_pd_data["reward"].append(reward)
        test_pd_data["steering_angle"].append(steering_angle)

        # get bezier curve data
        bez_point, _ =  env.closest_curve_point(env.cur_pos, env.cur_angle)

        # store bezier curve data
        bez_points["bez_x"].append(bez_point[0])
        bez_points["bez_z"].append(bez_point[2])

        # update display
        env.render()

        if done:
            if reward < 0:
                print(f"Test crashed, run again")
                env.reset()
                # reset data collection 
                # initialise test data for analysis
                test_pd_data = {
                    "cur_pos_x": [],
                    "cur_pos_z": [],
                    "distance_to_road_center": [],
                    "angle_from_straight_in_rads": [],
                    "reward": [],
                    "steering_angle": []
                }
                
                bez_points = {
                    "bez_x": [],
                    "bez_z": []
                }

                # initialise total reward
                total_reward = 0
                continue
            else:
                # successful run so save data to csv
                print(f"Test reached goal")
                save_data_to_csv(folder_name, "gt_udem1.csv", bez_points)
                save_data_to_csv(folder_name, f"test_pd_data.csv", test_pd_data)
                break

    

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Run Duckietown simulation and collect data')
    parser.add_argument('--folder_name', type=str, help='Folder name for simulation data')
    parser.add_argument('--map_name', type=str, help='Name of map to simulate in')
    args = parser.parse_args()

    main(args.folder_name, args.map_name)
