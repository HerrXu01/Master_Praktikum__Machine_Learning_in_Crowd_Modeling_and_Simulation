import torch
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import nn
from data_process.pedestrian_dataset import PedestrianDataset
from models.nn import NN
from tqdm import tqdm


def load_data(data_path:Path):
    """
    Loads data from a CSV file at the specified path.
    
    Parameters:
    - data_path (Path): The path to the CSV file containing the data.
    
    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays, one for the features and one for the targets.
    """
    df = pd.read_csv(data_path, header=0)
    data = df.values
    features = data[:, :-1]
    targets = data[:, -1]

    return features, targets


def split_data(features: np.ndarray, targets: np.ndarray, normal_train: bool, val_size=0.2, test_size=0.5):
    """
    Splits the data into training, validation, and test sets.
    
    Parameters:
    - features (np.ndarray): The features of the data.
    - targets (np.ndarray): The target values of the data.
    - normal_train (bool): A flag indicating whether to perform a normal train-test split or a different split.
    - val_size (float, optional): The proportion of the training data to be used as validation data.
    - test_size (float, optional): The proportion of the data to be used as test data.
    
    Returns:
    - Tuple: Depending on `normal_train`, either a 4-tuple (X_temp, X_test, y_temp, y_test) or a 6-tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_temp, X_test, y_temp, y_test = train_test_split(features, targets, test_size=test_size, random_state=42)
    if normal_train:
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        return  X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_temp, X_test, y_temp, y_test


def get_dataloader(*datasets, use_batch: bool, batch_size=64):
    """
    Creates DataLoader instances for the given datasets.
    
    Parameters:
    - datasets: Variable number of datasets, either 4 or 6 numpy arrays representing the features and targets for training, validation, and/or test sets.
    - use_batch (bool): A flag indicating whether to use batching.
    - batch_size (int, optional): The size of each batch.
    
    Returns:
    - Tuple[DataLoader, ...]: A tuple containing DataLoader instances for the provided datasets.
    """
    if len(datasets) == 4:
        X_train, X_test, y_train, y_test = datasets
        train_set = PedestrianDataset(X_train, y_train)
        test_set = PedestrianDataset(X_test, y_test)

        train_loader = DataLoader(train_set, batch_size=batch_size if use_batch else len(train_set), shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size if use_batch else len(test_set), shuffle=False)

        return train_loader, test_loader

    elif len(datasets) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = datasets
        train_set = PedestrianDataset(X_train, y_train)
        val_set = PedestrianDataset(X_val, y_val)
        test_set = PedestrianDataset(X_test, y_test)

        train_loader = DataLoader(train_set, batch_size=batch_size if use_batch else len(train_set), shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size if use_batch else len(val_set), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size if use_batch else len(test_set), shuffle=False)

        return train_loader, val_loader, test_loader
    
    else:
        raise ValueError("Invalid number of arguments. Expected 4 or 6.")


def normalize(x1, x2, x3=None):
    """
    Normalizes the given datasets using StandardScaler.
    
    Parameters:
    - x1 (np.ndarray): The training set to normalize.
    - x2 (np.ndarray): The validation or test set to normalize.
    - x3 (np.ndarray, optional): An optional third dataset to normalize.
    
    Returns:
    - Tuple[np.ndarray, ...]: A tuple of normalized datasets, corresponding to the input datasets.
    """
    scaler = StandardScaler()
    x1 = scaler.fit_transform(x1)
    x2 = scaler.transform(x2)
    if x3 is not None:
        x3 = scaler.transform(x3)
        return x1, x2, x3
    else:
        return x1, x2


def normal_train(config: dict):
    """
    Performs normal training using the configuration provided.
    
    Parameters:
    - config (dict): A dictionary containing configuration parameters for the training process.
    """
    train_data_path = Path(config["generated_data_dir"]) / config["train_data"]
    features, targets = load_data(train_data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets, normal_train=config["normal_train"], val_size=config["val_size"], test_size=config["test_size"])
    X_train, X_val, X_test = normalize(x1=X_train, x2=X_val, x3=X_test)
    train_loader, val_loader, test_loader = get_dataloader(X_train, X_val, X_test, y_train, y_val, y_test, use_batch=config["use_batch"], batch_size=config["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NN(config["model"]["num_layers"], config["model"]["num_nodes"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float('inf')
    best_epoch = 0

    wandb.init(
        project="Normal_training",
        entity  = "mlcms_group_e",
        config = config,
        name = "B180_epoch_100",
        save_code=True
    )

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        for features, targets in tqdm(train_loader):
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.array(train_loss).mean()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{config["epochs"]}:Train Loss: {train_loss}, Validation Loss: {val_loss}')

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    wandb.finish()
    print(f"The best epoch is {best_epoch}, with the best val loss {best_val_loss}.")


def cross_val_train(config: dict):
    """
    Performs training with cross-validation or on the full train set based on the configuration provided.
    
    Parameters:
    - config (dict): A dictionary containing configuration parameters for the training process.
    """
    if config["train_combine_RB"]:
        train_data_path = Path(config["generated_data_dir"]) / config["train_data"]
        features_0, targets_0 = load_data(train_data_path)
        X_train_0, X_test_0, y_train_0, y_test_0 = split_data(features_0, targets_0, normal_train=config["normal_train"], test_size=config["test_size"])
        train_data_path = Path(config["generated_data_dir"]) / config["train_data_1"]
        features_1, targets_1 = load_data(train_data_path)
        X_train_1, X_test_1, y_train_1, y_test_1 = split_data(features_1, targets_1, normal_train=config["normal_train"], test_size=config["test_size"])
        X_train = np.concatenate((X_train_0, X_train_1), axis=0)
        X_test = np.concatenate((X_test_0, X_test_1), axis=0)
        y_train = np.concatenate((y_train_0, y_train_1), axis=0)
        y_test = np.concatenate((y_test_0, y_test_1), axis=0)
    else:
        train_data_path = Path(config["generated_data_dir"]) / config["train_data"]
        features, targets = load_data(train_data_path)
        X_train, X_test, y_train, y_test = split_data(features, targets, normal_train=config["normal_train"], test_size=config["test_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config["output_filename"]
    
    if config["cross_val"]:
        kf = KFold(n_splits=config["fold_splits"], shuffle=True, random_state=42)
        wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"], config=config, save_code=True)

        print("Cross validation.")
        train_fold_loss = []
        val_fold_loss = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"Training fold {fold+1} ...")
            wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"], config=config, save_code=True, reinit=True)
            wandb.run.name = f"Fold_{fold+1}"
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            X_train_fold_scaled, X_val_fold_scaled = normalize(x1=X_train_fold, x2=X_val_fold)
            train_loader_fold, val_loader_fold = get_dataloader(X_train_fold_scaled, X_val_fold_scaled, y_train_fold, y_val_fold, use_batch=config["use_batch"], batch_size=config.get("batch_size", None))
            
            model = NN(config["model"]["num_layers"], config["model"]["num_nodes"]).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            for epoch in range(config["epochs"]):
                model.train()
                train_loss = []
                for features, targets in tqdm(train_loader_fold):
                    features, targets = features.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

                train_loss = np.array(train_loss).mean()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for features, targets in val_loader_fold:
                        features, targets = features.to(device), targets.to(device)
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader_fold)
                print(f'Fold {fold+1}/{config["fold_splits"]} Epoch {epoch+1}/{config["epochs"]}:Train Loss: {train_loss}, Validation Loss: {val_loss}')
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

                if epoch == (config["epochs"] - 1):
                    msg = f"Fold {fold}: Training loss {train_loss}, Validation loss {val_loss}\n"
                    with open(output_path, 'a') as file:
                        file.write(msg)
                
                    train_fold_loss.append(train_loss)
                    val_fold_loss.append(val_loss)
            
        train_fold_loss = np.array(train_fold_loss).mean()
        val_fold_loss = np.array(val_fold_loss).mean()
        msg = f"Cross validation result: Average training loss: {train_fold_loss}, average validation loss: {val_fold_loss}\n\n"
        with open(output_path, 'a') as file:
            file.write(msg)

    print("Training on the full train set ...")
    wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"], config=config, save_code=True, reinit=True)
    wandb.run.name = "Final_training_on_full_train_set"
    
    X_train_scaled, X_test_scaled = normalize(x1=X_train, x2=X_test)
    train_loader_full, test_loader_full = get_dataloader(X_train_scaled, X_test_scaled, y_train , y_test, use_batch=config["use_batch"], batch_size=config.get("batch_size", None))

    model = NN(config["model"]["num_layers"], config["model"]["num_nodes"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        for features, targets in tqdm(train_loader_full):
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.array(train_loss).mean()
        wandb.log({"train_loss": train_loss, "epoch": epoch})

    msg = f"Training on the full train set:\nTraining loss {train_loss}\n\n"
    with open(output_path, 'a') as file:
        file.write(msg)

    print("Testing on the other half of the same dataset ...")
    model.eval()
    test_loss_same = 0
    with torch.no_grad():
        for features, targets in test_loader_full:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss_same += loss.item()

    test_loss_same /= len(test_loader_full)
    wandb.log({"test_loss_on_same": test_loss_same})
    msg = "Testing on the other half of the same dataset " + config["train_data"] + "\n"
    msg += f"Test loss: {test_loss_same}\n\n"
    with open(output_path, 'a') as file:
        file.write(msg)

    if not config["train_combine_RB"]:
        print("Testing on the other dataset ...")
        test_other_path = Path(config["generated_data_dir"]) / config["test_on_other"]
        features_other, targets_other = load_data(test_other_path)
        _, X_test_other, _, y_test_other = split_data(features_other, targets_other, normal_train=config["normal_train"], test_size=config["test_size"])
        X_train_scaled, X_test_other_scaled = normalize(x1=X_train, x2=X_test_other)
        _, test_loader_other = get_dataloader(X_train_scaled, X_test_other_scaled, y_train, y_test_other, use_batch=config["use_batch"], batch_size=config.get("batch_size", None))

        model.eval()
        test_loss_other = 0
        with torch.no_grad():
            for features, targets in test_loader_other:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss_other += loss.item()

        test_loss_other /= len(test_loader_other)
        wandb.log({"test_loss_on_other": test_loss_other})
        msg = "Testing on the other dataset " + config["test_on_other"] + "\n"
        msg += f"Test loss: {test_loss_other}\n\n"
        with open(output_path, 'a') as file:
            file.write(msg)

    if config["train_combine_RB"]:
        print("Testing on half of the dataset ", config["train_data"])
        X_train_scaled, X_test_0_scaled = normalize(x1=X_train, x2=X_test_0)
        _, test_loader_0 = get_dataloader(X_train_scaled, X_test_0_scaled, y_train, y_test_0, use_batch=config["use_batch"], batch_size=config.get("batch_size", None))

        model.eval()
        test_loss_0 = 0
        with torch.no_grad():
            for features, targets in test_loader_0:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss_0 += loss.item()

        test_loss_0 /= len(test_loader_0)
        wandb.log({"test_loss_0": test_loss_0})
        msg = "Testing on half of the dataset " + config["train_data"] + "\n"
        msg += f"Test loss: {test_loss_0}\n\n"
        with open(output_path, 'a') as file:
            file.write(msg)

        print("Testing on half of the dataset ", config["train_data_1"])
        X_train_scaled, X_test_1_scaled = normalize(x1=X_train, x2=X_test_1)
        _, test_loader_1 = get_dataloader(X_train_scaled, X_test_1_scaled, y_train, y_test_1, use_batch=config["use_batch"], batch_size=config.get("batch_size", None))

        model.eval()
        test_loss_1 = 0
        with torch.no_grad():
            for features, targets in test_loader_1:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss_1 += loss.item()

        test_loss_1 /= len(test_loader_1)
        wandb.log({"test_loss_1": test_loss_1})
        msg = "Testing on half of the dataset " + config["train_data_1"] + "\n"
        msg += f"Test loss: {test_loss_1}\n\n"
        with open(output_path, 'a') as file:
            file.write(msg)

    wandb.finish()
