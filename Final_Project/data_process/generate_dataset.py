import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from tqdm import tqdm

def save_data(frame_samples: np.ndarray, data_dir: Path, data_filename: str):
    num_col = frame_samples.shape[1]
    data = np.copy(frame_samples)

    column_names = ["sk"]  # the first feature is the mean spacing
    for i in range(num_col-2):
        column_names.append(f"feature{i+1}")  # the coordinates of the K neighbors
    column_names.append("speed")

    df = pd.DataFrame(data, columns=column_names)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / data_filename
    df.to_csv(data_path, index=False)


def generate_dataset(filepath: str, K: int, data_dir: Path, data_filename: str):

    print("Generating dataset.")

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

    frame_samples = np.vstack(frame_samples)

    save_data(frame_samples=frame_samples, data_dir=data_dir, data_filename=data_filename)

    print("Dataset generated successfully!")
    print("The shape of the dataset (containing features and targets) is ", frame_samples.shape, ".")

    return frame_samples
