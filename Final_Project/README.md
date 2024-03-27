# Final Project of MLCMS

In this project, we implement the Fundamental Diagram based Model and Neural Network Model for predicting the speeds of pedestrians based on the paper ["Prediction of Pedestrian Speed with Artificial Neural Networks"](https://arxiv.org/abs/1801.09782) by A. Tordeux, M. Chraibi, A. Seyfried, and A. Schadschneider.

## Environment Setup

* We recommend using `conda` to install dependencies and set up the environment.  

* After installing `conda`, run 
```
conda install mamba -c conda-forge
mamba env create -f environment.yaml
```

* Run below code to activate the environment
```
conda activate mlcms
```


## Training NN

If training NN, remember to specify the argument `config`. If it is the first time training, remember to specify the argument `raw_data_dir`.  

Here is an example for the first time training NN on dataset B. Run
```
python main.py --config "configs/train_B.yaml" --raw_data_dir "raw_data"
```