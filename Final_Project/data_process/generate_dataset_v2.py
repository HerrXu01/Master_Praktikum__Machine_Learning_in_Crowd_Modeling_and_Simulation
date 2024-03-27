import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from tqdm import tqdm

def save_data(samples: np.ndarray, data_dir: Path, data_filename: str):
    """
    Save the given samples to a CSV file in the specified directory with the given filename.

    Parameters:
    - samples (np.ndarray): A 2D numpy array where each row represents a sample and each column a feature.
    - data_dir (Path): The directory path where the CSV file will be saved. If the directory does not exist, it will be created.
    - data_filename (str): The name of the CSV file to save the data to.

    The function creates a pandas DataFrame from the numpy array, with the first column named "sk" representing the mean spacing,
    followed by columns named "feature1", "feature2", etc., for each feature, and the last column named "speed".
    The DataFrame is then saved to a CSV file in the specified directory.
    """
    num_col = samples.shape[1]
    data = np.copy(samples)

    column_names = ["sk"]  # the first feature is the mean spacing
    for i in range(num_col-2):
        column_names.append(f"feature{i+1}")  # the coordinates of the K neighbors
    column_names.append("speed")

    df = pd.DataFrame(data, columns=column_names)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / data_filename
    df.to_csv(data_path, index=False)


def generate_dataset(exp: str, filepaths: list, K: int, data_dir: Path, data_filenames: list):
    """
    Generate a dataset by processing raw data files specified in `filepaths`, finding K nearest neighbors for points in each frame,
    and saving the processed data to CSV files in the specified directory.

    Parameters:
    - exp (str): A string identifier for the experiment or dataset generation run. It is used for logging purposes.
    - filepaths (list): A list of file paths, each pointing to a raw data file to be processed.
    - K (int): The number of nearest neighbors to find for each point in the dataset.
    - data_dir (Path): The directory where the processed data files will be saved. If it does not exist, it will be created.
    - data_filenames (list): A list of strings representing the filenames for the processed data CSV files. This list should have the same length as `filepaths`.

    For each file in `filepaths`, the function reads the data, computes the speed for each point based on its movement between frames,
    finds the K nearest neighbors for each point in each frame, and calculates features based on these neighbors.
    The processed data for each file is then saved to a new CSV file in `data_dir` with the corresponding name from `data_filenames`.

    The function logs the start and completion of the dataset generation process, including the directory where the dataset is saved.
    """

    print("Generating dataset ", exp,".")

    for i, filepath in enumerate(filepaths):

        raw_data = pd.read_csv(filepath, sep=' ', header=None)
        raw_data.columns = ['ID', 'FRAME', 'X', 'Y', 'Z']
        raw_data = raw_data.drop(columns='Z')

        # Add speed
        df_groupbyID = {pid: ped for (pid, ped) in raw_data.groupby("ID")}
        for ped in df_groupbyID.values():
            coor_diff = ped[['X', 'Y']].diff()
            distances = np.sqrt(coor_diff['X']**2 + coor_diff['Y']**2)
            distances.iloc[0] = distances.iloc[1]  # Let the first and second frame have the same speed
            ped["speed"] = distances * 0.16  # The unit is m/s

        df_with_speed = pd.concat(df_groupbyID.values(), ignore_index=True)
        df_groupbyF = {fra: ped for (fra, ped) in df_with_speed.groupby("FRAME")}

        frame_samples = []

        # find nearest neighbors
        for frame in tqdm(df_groupbyF.values()):
            if frame.shape[0] <= K:
                continue

            points = frame[['X', 'Y']].values
            nbrs = NearestNeighbors(n_neighbors=K+1, metric='euclidean').fit(points)
            k_distances, k_indices = nbrs.kneighbors(points)
            nearest_diffs = np.array([points[k_indices[i]][1:] - points[i] for i in range(len(points))])
            flattened_diffs = nearest_diffs.reshape(nearest_diffs.shape[0], -1)
            sk = k_distances.sum(axis=1) / K  # the mean spacing
            features = np.concatenate((sk.reshape(-1, 1), flattened_diffs), axis=1)
            frame_samples.append(np.concatenate((features, frame[['speed']].values), axis=1))

        if len(frame_samples) == 0:
            continue
        else:
            frame_samples = np.vstack(frame_samples)
            save_data(samples=frame_samples, data_dir=data_dir, data_filename=data_filenames[i])

    print("Dataset ", exp, " generated successfully!")
    print("The dataset can be found at ", data_dir, ".")
