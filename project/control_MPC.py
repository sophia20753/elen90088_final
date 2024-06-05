#!/usr/bin/env python3

# import relevant libraries
from pickle import FALSE, TRUE
from PIL import Image
import time
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv

import numpy as np
import csv

import do_mpc
from do_mpc.data import save_results, load_results
import numpy as np
import math as math

def get_dir_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])

def main():
    csvfile = open('MPC_data_straight_road.csv', 'w', newline='')
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    header = ['Curr Pos X','Curr Pos Y','Curr Ang',                       # Current position data
              'Position error', 'Angle Error',                            # Error relative to Bezier Curve
              'Ang Vel (control)', 'Curr Reward', 'Total Reward',         # Control action and rewards
              'Closest Curve X','Closest Curve Y', 'Bezier Tangent',      # 'Optimal' closest Bezier point
              'Step number', 'Iteration runtime', 'Total Script runtime'] # Temporal data
    csv_writer.writerow(header)     # write the header

    # Add to DuckietownEnv object constructor to specify starting tile:
    # user_tile_start=(3,4)
    
    # create environment with a map
    """ 
    Different map types: 
    4way, loop_dyn_duckiebots, loop_empty, loop_obstacles, loop_pedestrians, regress_4way_adam, regress_4way_drivable, small_loop,
    small_loop_cw, straight_road, udem1, zigzag_dists
    """

    env = DuckietownEnv(map_name='straight_road',domain_rand=False, draw_bbox=False, draw_curve = True, max_steps = 7000)
    env.reset()
    env.render()

    start_pos = env.cur_pos
    # define total reward
    total_reward = 0    

    # Things that can be initialised outside of the loop
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    # State variable description
    X_1 = model.set_variable(var_type='_x', var_name='X_1', shape=(1,1))    # x_n
    X_2 = model.set_variable(var_type='_x', var_name='X_2', shape=(1,1))    # y_n
    X_3 = model.set_variable(var_type='_x', var_name='X_3', shape=(1,1))    # phi_n

    X_4 = model.set_variable(var_type='_x', var_name='X_4', shape=(1,1))    # x_n+1
    X_5 = model.set_variable(var_type='_x', var_name='X_5', shape=(1,1))    # y_n+1
    X_6 = model.set_variable(var_type='_x', var_name='X_6', shape=(1,1))    # phi_n+1

    LP_DIST = model.set_variable(var_type='_p', var_name='LP_DIST', shape=(1,1)) 
    TANG_X  = model.set_variable(var_type='_p',var_name='TANG_X', shape=(1,1))
    TANG_Y  = model.set_variable(var_type='_p',var_name='TANG_Y', shape=(1,1))
    PERP_X  = model.set_variable(var_type='_p',var_name='PERP_X', shape=(1,1))
    PERP_Y  = model.set_variable(var_type='_p',var_name='PERP_Y', shape=(1,1))
    X0      = model.set_variable(var_type='_p',var_name='X0', shape=(1,1))
    Y0      = model.set_variable(var_type='_p',var_name='Y0', shape=(1,1))

    # Control variable description
    # car_speed = model.set_variable(var_type='_u', var_name='phi_m_1_set')    #Floored to a constant to reduce complexity
    w = model.set_variable(var_type='_u', var_name='w')   # Angular speed


    # Next we should define uncertain parameters - not needed for simulation (Duckietown)
    # => Hardcode all constant system parameters
    FPS = 30
    del_t = 1/FPS    # Sampling time is inverse of frame rate
    car_speed = 0.1  # In the duckietown code, a car_speed of 1 corresponds to a simulation velocity of 0.6 m/s
    K = 0.1          # The conversion from control input to in-simulation angular speed

    # Define RHS i.e. system dynamics
    # => state-space model
    model.set_rhs('X_1', X_4)
    model.set_rhs('X_2', X_5)
    model.set_rhs('X_3', X_6)

    model.set_rhs('X_4', X_1 + np.cos(X_3) * car_speed * 0.6 * del_t)
    model.set_rhs('X_5', X_2 - np.sin(X_3) * car_speed * 0.6 * del_t)
    model.set_rhs('X_6', X_3 + K * w)
    
    model.setup()

    # End of Model Setup
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Configure MPC controller
    
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 5,
        't_step': del_t,
        'n_robust': 1,
        'store_full_solution': False,
    }
    mpc.set_param(**setup_mpc)

    mterm = X_1 - X_1   # "Meyer term" refers to the terminal constraint. For now let this be zero
    lterm = 50 * (LP_DIST - np.dot([X_1 - X0, 0, X_2 - Y0], [PERP_X, 0, PERP_Y]))**4 + 6 * (np.mod(np.arctan2(-1 * TANG_Y, TANG_X) - X_3, 2*np.pi))**4
    mpc.set_rterm(    # This sets the R matrix in the do-mpc cost formulation. Is specific to an MPC implementation
        w = 1e-4,     # Applies a cost to 'smoothen' the control sequence -> is independant of Duckietown rewards
    )
    
    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.bounds['lower','_u', 'w'] = -10.0
    mpc.bounds['upper','_u', 'w'] =  10.0
    
    mpc.set_uncertainty_values(LP_DIST = [0], TANG_X = [0], TANG_Y = [0], PERP_X = [0],        # Initialise the 'uncertainty parameters'
                                    PERP_Y = [0], X0 = [0], Y0 = [0])
    
    mpc.settings.supress_ipopt_output()       # Stop extraneous computation for printing optimisation output
    mpc.setup()                               # setup() needs to be called to compile this initial controller -> otherwise we get errors

    # main loop
    simcontinue = True
    start = time.time()
    while simcontinue:
        start_it = time.time()
        # Get current point data
        cur_pos = env.cur_pos
        x0 = cur_pos[0]
        y0 = cur_pos[2]
        cur_angle = env.cur_angle
        dir_vec = get_dir_vec(cur_angle)   # Unit heading direction vector
        
        # Obtain error quantities
        closest_point, point_tangent = env.closest_curve_point(cur_pos, cur_angle)    # Note point_tangent is a unit vector => magnitude = 1
        perp_vec = [1.0 * point_tangent[2], 0, -1.0 * point_tangent[0]]               # Perpendicular vector to tangent of closest point
        lp = env.get_lane_pos2(cur_pos,cur_angle)
        dot_prod = np.dot(point_tangent, dir_vec)
        pos_err = lp.dist                               # Distance from road centre (Bezier point) - *Signed Value*
        ang_err = np.arccos(dot_prod)                   # This is absolute angle error
        if cur_angle < 0:
            ang_err = -ang_err                          # Sign the angle error
        
        # Each iteration we need to update the model parameters
        mpc.set_uncertainty_values(LP_DIST = [lp.dist], TANG_X = [point_tangent[0]], TANG_Y = [point_tangent[2]], PERP_X = [perp_vec[0]],
                                    PERP_Y = [perp_vec[2]], X0 = [x0], Y0 = [y0])
        
        init_state = np.array([x0, y0, cur_angle, x0, y0, cur_angle])
        mpc.set_initial_guess()
        w_next = mpc.make_step(init_state)              # Perform optimisation and assign the first control effort to the next actuated control effort

        obs, reward, done, info = env.step([car_speed, w_next])    #Actuate the robot
        print(f"Distance Error: {pos_err}, Angle Error: {np.rad2deg(ang_err)} degrees")

        # update total reward
        total_reward += reward

        end_it = time.time()

        """
        Recall:
        header = ['Curr Pos X','Curr Pos Y','Curr Ang',                   
              'Position error', 'Angle Error',                           
              'Ang Vel (control)', 'Curr Reward', 'Total Reward',        
              'Closest Curve X','Closest Curve Y', 'Bezier Tangent',      
              'Step number', 'Iteration runtime', 'Total Script runtime']
        """
        # Write data to csv file
        csv_writer.writerow([x0, y0,np.rad2deg(cur_angle),
                             pos_err,ang_err,                               
                             w_next,reward,total_reward,                    
                        closest_point[0],closest_point[2],point_tangent,
                            env.step_count, end_it - start_it, end_it - start])   

        if env.step_count > 100: 
            distance = math.sqrt((cur_pos[0] - start_pos[0])**2 + (cur_pos[2] - start_pos[2])**2)
            if distance <= 0.05:
                simcontinue = False

        # let the simulator update its display
        env.render()

        if done:
            if reward < 0:
                print("*** CRASHED ***")
            print("Exiting")
            print(info)
            break


if __name__  == '__main__':
    main()
