'''
Script Name: project_functions.py

Description:
    Module of helper functions for ML project.

Author:
    Sophia Chiang
    June 12th 2024
'''

# import relevant libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def save_data_to_csv(foldername, filename, data):
    '''Save data from dict to a .csv file'''
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


def interpolate_trajectory(indices, x, y, new_indices):
    """Interpolate x and y coordinates to new indices"""
    interpolated_x = np.interp(new_indices, indices, x)
    interpolated_y = np.interp(new_indices, indices, y)
    return np.vstack((interpolated_x, interpolated_y)).T


def huber_loss_coordinates(xtrue, ytrue, xpred, ypred, delta=1.2, plot=False):
    """Calculate the Huber loss between two trajectories after aligning them using DTW"""
    # Create implicit indices based on the length of the trajectories
    indices1 = np.arange(len(xtrue))
    indices2 = np.arange(len(xpred))

    # Determine new common indices
    common_indices = np.linspace(0, max(len(xtrue), len(xpred)) - 1, num=100)

    # Interpolate trajectories
    trajectory1_interp = interpolate_trajectory(indices1, xtrue, ytrue, common_indices)
    trajectory2_interp = interpolate_trajectory(indices2, xpred, ypred, common_indices)

    # Compute DTW alignment
    distance, path = fastdtw(trajectory1_interp, trajectory2_interp, dist=euclidean)

    # Align trajectories using DTW path
    aligned_trajectory1 = np.array([trajectory1_interp[i] for i, _ in path])
    aligned_trajectory2 = np.array([trajectory2_interp[j] for _, j in path])

    # Compute Huber loss on aligned trajectories
    loss = huber_loss(aligned_trajectory1, aligned_trajectory2, delta)

    if plot:
        # Plotting for visualization
        plt.plot(xtrue[::50], ytrue[::50], 'bo-', label='Ground Truth')
        plt.plot(xpred[::50], ypred[::50], 'ro-', label='Model')
        plt.plot(aligned_trajectory1[:, 0], aligned_trajectory1[:, 1], 'b--', alpha=0.5, label='Aligned Ground Truth Trajectory')
        plt.plot(aligned_trajectory2[:, 0], aligned_trajectory2[:, 1], 'r--', alpha=0.5, label='Aligned Model Trajectory')
        plt.legend()
        plt.title('Trajectories and DTW Alignment')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    return loss


def get_lap(x, y, tol_x, tol_y):
    '''Extract data for one lap'''

    # Get end points
    end_x = x[-1]
    end_y = y[-1]

    # Find when x and y matches end_x and end_y
    lap_start_index = None

    # Start looking from the end of data
    for i in range(2000, -1, -1):
        if (abs(x[i] - end_x) <= tol_x) and (abs(y[i] - end_y) <= tol_y):
            lap_start_index = i
            break

    if lap_start_index is not None:
        x_lap = x[lap_start_index:]
        y_lap = y[lap_start_index:]
    else:
        x_lap = x
        y_lap = y

    return x_lap, y_lap


def interpolate_trajectories(x_gt, y_gt, x_pd, y_pd, x_model, y_model):
    '''Interpolate ground truth and PD trajectories to match model's trajectory length'''
    # Interpolate ground truth trajectory
    f_x_gt = interp1d(np.linspace(0, 1, len(x_gt)), x_gt, kind='linear')
    f_y_gt = interp1d(np.linspace(0, 1, len(y_gt)), y_gt, kind='linear')

    # Interpolate PD trajectory
    f_x_pd = interp1d(np.linspace(0, 1, len(x_pd)), x_pd, kind='linear')
    f_y_pd = interp1d(np.linspace(0, 1, len(y_pd)), y_pd, kind='linear')

    # Match ground truth and PD trajectory lengths with model trajectory length
    x_gt_interp = f_x_gt(np.linspace(0, 1, len(x_model)))
    y_gt_interp = f_y_gt(np.linspace(0, 1, len(y_model)))

    x_pd_interp = f_x_pd(np.linspace(0, 1, len(x_model)))
    y_pd_interp = f_y_pd(np.linspace(0, 1, len(y_model)))

    return x_gt_interp, y_gt_interp, x_pd_interp, y_pd_interp


def print_huber_losses(ytrue, ypred, x_pd, y_pd, x_gt, y_gt, x_model, y_model):
    '''Calculate and print losses'''
    # Calculate model test loss
    model_test_hl = huber_loss(ypred, ytrue)

    # Calculate PD and model test loss
    pd_xy_model_test_hl = huber_loss_coordinates(x_pd, y_pd, x_model, y_model)

    # Calculate ground truth and model test loss
    gt_xy_model_test_hl = huber_loss_coordinates(x_gt, y_gt, x_model, y_model)

    # Print losses
    print(f"Model Test Huber Loss: {model_test_hl}")
    print(f"PD and Model Huber Loss: {pd_xy_model_test_hl}")
    print(f"Ground Truth and Model Huber Loss: {gt_xy_model_test_hl}")


def plot_traj_reward(x_pd_lap, y_pd_lap, x_model_lap, y_model_lap, x_gt_lap, y_gt_lap, df_model, df_pd, title):
    '''Plot trajectory and reward'''
    # Create subplots
    #fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5,8))
    fig, ax1 = plt.subplots(figsize=(5,4))

    #fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot trajectories for qualitative comparison
    ax1.scatter(x_pd_lap[::50], y_pd_lap[::50], marker='x', label='PD', color='C0', s=25)
    ax1.plot(x_model_lap, y_model_lap, label='Model', color='C1')
    ax1.plot(x_gt_lap, y_gt_lap, label='Ground Truth', color='C2', alpha=0.5)
    ax1.legend()
    #ax1.set_title("Trajectory", fontsize=12, fontstyle='italic')
    '''
    # plot reward values for dnn and pd 
    reward_model = df_model["reward"].values
    reward_pd = df_pd["reward"].values

    timestep_model = range(len(df_model))
    timestep_pd = range(len(df_pd))

    ax2.plot(timestep_pd, reward_pd, label='PD')
    ax2.plot(timestep_model, reward_model, label='Model')
    ax2.legend()
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward", fontsize=12, fontstyle='italic')
    '''
    plt.tight_layout()

    plt.show()


def plot_misalignment(x_model, y_model, x_gt_lap, y_gt_lap, angle_tol=10, x_tol=0.7, y_tol=0.1):
    '''Plot misalignment using angle and distance deviation'''

    # Interpolate ground truth to match the length of model trajectory
    f_x = interp1d(np.linspace(0, 1, len(x_gt_lap)), x_gt_lap, kind='linear')
    f_y = interp1d(np.linspace(0, 1, len(y_gt_lap)), y_gt_lap, kind='linear')
    x_gt_lap = f_x(np.linspace(0, 1, len(x_model)))
    y_gt_lap = f_y(np.linspace(0, 1, len(y_model)))

    # Calculate direction vectors
    def calculate_direction_vectors(x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        return dx, dy

    dx_model, dy_model = calculate_direction_vectors(x_model, y_model)
    dx_gt, dy_gt = calculate_direction_vectors(x_gt_lap, y_gt_lap)

    # Calculate angles between direction vectors
    def calculate_angles(dx1, dy1, dx2, dy2):
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = np.sqrt(dx1**2 + dy1**2)
        magnitude2 = np.sqrt(dx2**2 + dy2**2)
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical issues
        return np.degrees(angle)

    angles = calculate_angles(dx_model, dy_model, dx_gt, dy_gt)

    # Calculate distances between corresponding points
    distances_x = np.abs(x_model - x_gt_lap)
    distances_y = np.abs(y_model - y_gt_lap)

    # Initialize variables
    deviation_points = []

    # Find points where angle or distance deviation exceeds thresholds
    for i in range(len(angles)):
        if angles[i] > angle_tol or (distances_x[i] > x_tol and distances_y[i] > y_tol):
            deviation_points.append((x_model[i+1], y_model[i+1]))  # i+1 because we used np.diff

    # Plot trajectory diagram with deviation points marked
    plt.plot(x_model, y_model, label='Model Trajectory')
    plt.plot(x_gt_lap, y_gt_lap, label='Ground Truth Trajectory')
    deviation_points = np.array(deviation_points)
    if deviation_points.size > 0:
        plt.scatter(deviation_points[:, 0], deviation_points[:, 1], color='red', label='Deviation Points', s=7.5)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory with Deviation Points Marked')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Time Spent Misaligned: {len(deviation_points)}")