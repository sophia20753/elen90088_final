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

def main(num_simulations, map_name):
    # create environment with training map
    env = DuckietownEnv(map_name=map_name, domain_rand=False, draw_bbox=False)

    foldername = f"train_data_{map_name}"

    # create main folder for training data
    if not os.path.exists("train_data"):
        os.makedirs("train_data")

    # create subfolder for map data
    subfolder_path = os.path.join("train_data", foldername)
    if not os.path.exists(subfolder_path):
        os.makedirs(foldername)

    for sim in range(num_simulations):
        env.reset()
        env.render()

        total_reward = 0

        # initialise data dict for this run 
        sim_data =  {
            "timestep": [],
            "car_speed": [],
            "steering_angle": [],
            "distance_to_road_center": [],
            "angle_from_straight_in_rads": [],
            "reward": [],
            "total_reward": []
        }

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
            sim_data["timestep"].append(env.step_count)
            sim_data["car_speed"].append(car_speed)
            sim_data["steering_angle"].append(steering_angle)
            sim_data["distance_to_road_center"].append(distance_to_road_center)
            sim_data["angle_from_straight_in_rads"].append(angle_from_straight_in_rads)
            sim_data["reward"].append(reward)
            sim_data["total_reward"].append(total_reward)

            # update display
            env.render()

            if done:
                if reward < 0:
                    print(f"Simulation {sim}: Crashed")
                    # run again
                    env.reset()
                    total_reward = 0

                    # initialise data dict for this run 
                    sim_data =  {
                        "timestep": [],
                        "car_speed": [],
                        "steering_angle": [],
                        "distance_to_road_center": [],
                        "angle_from_straight_in_rads": [],
                        "reward": [],
                        "total_reward": []
                    }
                    continue
                else:
                    # successful run so save data to csv
                    print(f"Simulation {sim}: Reached goal")
                    save_data_to_csv(foldername, f"training_run{sim}.csv", sim_data)
                break
    env.close()

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Run Duckietown simulation and collect data')
    parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--map_name', type=str, help='Name of map to simulate in')
    args = parser.parse_args()

    main(args.num_simulations, args.map_name)
