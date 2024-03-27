import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
import os

import warnings
warnings.filterwarnings('ignore')

def load_scenario(geometry, tail):
    
    """
    Reads the given file and names the columns to create a pandas Dataframe.
    
    Args:
        
        - geometry (str): data configuration name (whether Corridor or Bottleneck).
        - tail (str): specific file name. 
    
    Returns:
    
        - pd.DataFrame: prepared DataFrame.
    """

    file_path = f"raw_data/{geometry}/{tail}.txt"
    df = pd.read_csv(file_path, delimiter = ' ')
    df.columns = ['id', 'frame', 'x', 'y', 'z']
    
    return df


def get_pedestrian_data(pedestrian_id, configuration_df):
        
    """
    Extracts information of a pedestrian for a particular configuration.
    
    Args:
        
        - pedestrian_id (int): id of pedestrian.
        - configuration_df (pd.DataFrame): configuration dataframe. 
    
    Returns:
    
        - pd.DataFrame: filtered DataFrame with frame number and coordinate values of the required pedestrian.
    """

    df = configuration_df[configuration_df['id'] == pedestrian_id]
    # removing redundant information
    filtered_df = df[['frame','x','y','z']]
    
    return filtered_df 

def visualize_paths(lst_pedestrian_id, configuration_df):
        
    """
    Plots the trajectories of all pedestrians given in the list and present in the given configuration.
    
    Args:
        
        - lst_pedestrian_id (list): list of pedestrian ids.
        - configuration_df (pd.DataFrame): configuration dataframe. 
    
    Returns:
    
        - None.
    """
    
    # warning, if the lst_pedestrian_id is empty. 
    if len(lst_pedestrian_id) == 0:
        print("NO PEDESTRIANS FOUND")
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in lst_pedestrian_id:
        data = get_pedestrian_data(i, configuration_df)
        ax.plot(data['x'],data['y'], data['z'], label=f'Pedestrian {i}')
       
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title("Trajectory Plot")
    
    ax.legend()
    
    return
    
def model(params, s):
            
    """
    Gives the predicted values for a given set of params
    
    Args:
        
        - params (list): list of parameters [l, v0, T].
        - s (np.ndarray): sk values. 
    
    Returns:
    
        - np.ndarray: values of velocity, predicted by our model.
    """
    
    l, v0, T = params
    return v0 * (1 - np.exp((l - s) / v0 * T))

def loss(params, s, v):
             
    """
    Gives the mean squared error loss between predicted velocity values and avtual velocity values.
    
    Args:
        
        - params (list): list of parameters [l, v0, T].
        - s (np.ndarray): sk values. 
        - v (np.ndarray): actual velocity values
    
    Returns:
    
        - np.ndarray: MSE loss.
    """
    
    v_pred = model(params, s)
    return np.mean((v - v_pred) ** 2)

def CSV_merger(file_1, file_2):
    
    """
    Takes in two CSV files as input, merges them and saves them as a new file.
    
    Args:
        
        - file_1 (str): file 1 name.
        - file_2 (str): file 2 name. 
    
    Returns:
    
        - None.
    """
    
    file_1_data = pd.read_csv(f"generated_data/K_10/{file_1}.csv")
    file_2_data = pd.read_csv(f"generated_data/K_10/{file_2}.csv")

    merged_data = file_1_data.append(file_2_data, ignore_index=True)
    
    new_file_path = f"generated_data/K_10/{file_1}+{file_2}.csv"
    
    # prevents re-creation of file, if the file already exists.
    if not os.path.exists(new_file_path):
        merged_data.to_csv(new_file_path, index=False)
        print(f"Merged file saved as {new_file_path}")
    else:
        print(f"The file {new_file_path} already exists. Not performing the merge.")
        
    return
        
def column_extractor(file):
    
    """
    Takes in input as a file and returns the sk and actual_v values. 
    
    Args:
        
        - file (str): file name.
    
    Returns:
    
        - np.ndarray: mean spacing sk values.
        - np.ndarray: actual velocity values.
    """
    
    data = pd.read_csv(f"generated_data/K_10/{file}.csv")
    data = data[["sk","speed"]].to_numpy()
    sk = data[:, 0]/100
    actual_v = data[:, 1]
    
    return sk, actual_v

def optimal_parameter_calculator(train_file, test_file):
        
    """
    Takes in train file and test file and calculates the best parameters based on train data.
    
    Args:
        
        - train_file (str): train file name.
        - test_file (str): test file name.
    
    Returns:
    
        - list: array of optimal parameters [l, v0, T].
        - str: string label to be returned for graphing purposes.
        - float: actual velocity values.
    """

    sk_train, actual_v_train = column_extractor(train_file)
    sk_test, actual_v_test = column_extractor(test_file)
    
    # defining bounds for the operator.
    bounds = [(0, 100), (0, 100), (0, 100)]
    
    # using differential evolution to find global minima.
    result = differential_evolution(loss, bounds, args=(sk_train, actual_v_train))
    
    # extracting optimal parameters
    l_opt, v0_opt, T_opt = result.x
    
    num = "R+B"
    denum = "R+B"
    
    print(f'Optimal parameters: l = {l_opt}, v0 = {v0_opt}, T = {T_opt}')
    
    if len(train_file) == 5 & len(test_file) == 5:
        num = train_file[0]
        denum = test_file[0]
    elif len(train_file) != 5 & len(test_file) == 5:
        denum = test_file[0]
    elif len(train_file) == 5 & len(test_file) != 5:
        num = train_file[0]
    
    optimal_parameters = [l_opt, v0_opt, T_opt]
    final_loss = loss(optimal_parameters, sk_test, actual_v_test)
    
    print(f'Final loss of {num}/{denum}: {final_loss}')
    print()
    
    return optimal_parameters, f"{num}/{denum}", final_loss
