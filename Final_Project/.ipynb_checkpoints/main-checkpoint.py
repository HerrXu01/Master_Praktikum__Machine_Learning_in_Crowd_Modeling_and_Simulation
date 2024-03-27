import yaml
import argparse
from pathlib import Path
from data_process.generate_dataset_v2 import generate_dataset

# Set up argument parser
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction of Pedestrian Speed with NN')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--raw_data_dir', type=str, default=None, help='Path to the raw data, only needs to be specified at the first time')

    # Parse arguments
    args = parser.parse_args()

    # Load data from YAML file
    with open(Path(args.config), 'r') as file:
        config = yaml.safe_load(file)

    raw_data_dir = Path(args.raw_data_dir)
    generated_data_dir = Path(config["generated_data_dir"])
    if not generated_data_dir.exists():
        generated_data_dir.mkdir(parents=True, exist_ok=True)

        bottleneck_dir = raw_data_dir / "Bottleneck_Data"
        corridor_dir = raw_data_dir / "Corridor_Data"
        bottleneck_files = list(bottleneck_dir.glob('**/*.txt'))
        B_filenames = ["B_070.csv", "B_095.csv", "B_120.csv", "B_180.csv"]
        corridor_files = list(corridor_dir.glob('**/*.txt'))
        R_filenames = ["R_015.csv", "R_030.csv", "R_060.csv", "R_085.csv", "R_095.csv", "R_110.csv", "R_140.csv", "R_230.csv"]
        generate_dataset("B", bottleneck_files, config["K"], generated_data_dir, B_filenames)
        generate_dataset("R", corridor_files, config["K"], generated_data_dir, R_filenames)

    assert any(generated_data_dir.iterdir()), "The generated data directory is empty! Please check the code."

    