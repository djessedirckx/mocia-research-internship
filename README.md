# Match-Net Tensorflow2 implementation

## About
This repository contains a Tensorflow2 implementation of the deep neural network *Match-Net* for survival prediction. This method is described in the paper "*Dynamic Prediction in Clinical Survival Analysis Using Temporal Convolutional Networks*" [1]. The purpose of this repository is to implement the described network and to reproduce results presented in this paper.

[1] [D. Jarrett, J. Yoon and M. van der Schaar, "Dynamic Prediction in Clinical Survival Analysis Using Temporal Convolutional Networks," in IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 2, pp. 424-436, Feb. 2020, doi: 10.1109/JBHI.2019.2929264](https://ieeexplore.ieee.org/abstract/document/8765241?casa_token=x4POn0uI8ngAAAAA:RBVJpEukjCDBpauUNJFpax_bndPbXmuWnx4E9QJ_Y3-vDaQ3S23j4aHVWZSo7Kwi51Eil-BB8D2nEr4).

## Requirements & installation
All code in this repository is written using the Python3 programming language. To be able to succesfully use the source code, the following dependencies are required:
- Python 3 (during development, version 3.9.4 was used. Other Python 3 versions might work as well, but it is likely that other version of dependencies listed in `requirements.txt` are needed)
- [Tadpole dataset](https://tadpole.grand-challenge.org/Data/). Store the data in a directory called `tadpole_challenge` in the root of the project folder.

The source makes used of the dependencies listed in `requirements.txt`. The dependencies can be installed using `pip`, using the following command (preferably using a virtual environment):
```
pip install -r requirements.txt
```

## Usage
The software can be used for 2 different purposes. It can be used for hyperparameter tuning, to find the best-performing set of hyperparameters for a given prediction horizon. It can also be used to train the network, using a predefined set of parameters.

### Hyperparameter tuning
The hyperparameters for a given prediction horizon can be tuned using the script defined in `tune_model.py`. Below, possible command line arguments for defining behaviour are listed. `--label_forwarding`, `--weight_regularisation` & `--oversampling` works as a boolean. Including them as a command line argument will activate the respective functionality.

```
usage: tune_model.py [-h] [--prediction_horizon] [--cross_val_splits] [--max_trials] [--label_forwarding] [--weight_regularisation] [--oversampling]

Tune MATCH-net hyperparameters for a desired prediction horizon

optional arguments:
  -h, --help                show this help message and exit

  --prediction_horizon      Number of events in the future to predict (int)
                            
  --cross_val_splits        Number of cross validation splits to use (int)
                            
  --max_trials              Max number of trials to perform search (int)
                            
  --label_forwarding        Employ label forwarding to passively increase amount of positive labels

  --weight_regularisation   Use weight regularisation in model
                            
  --oversampling            Apply oversampling of the minority class in the training data
```

### Model training
The model can be trained using a predefined set of parameters as well using the `train_model.py` script.

```
usage: train_model.py [-h] [--eval_time]

Train Match-Net for a desired prediction horizon

optional arguments:
  -h, --help            show this help message and exit
  --eval_time           Brier evaluation time (int)
```

The desired model parameters can be defined using a `MatchNetConfig` instance. An example is shown below for prediction horizon 1. This should be defined at the bottom of `train_model.py`.

```python
matchnet_config = MatchNetConfig(
    pred_horizon=1,
    window_length=5,
    cov_filters=512,
    mask_filters=8,
    cov_filter_size=6,
    mask_filter_size=6,
    cov_input_features=35,
    mask_input_features=35,
    dense_units=256,
    conv_blocks=1,
    dense_layers=1,
    dropout_rate=0.1,
    val_frequency=7,
    label_fowarding=True,
    weight_regularisation=False,
    oversampling=True,
    oversample_ratio=0.33,
    learning_rate=0.0001,
    output_path="output/oversampl_forwarding")
```

## Files
This repository contains multiple files. Below, the file structure is shown and each file is shortly introduced. All files contain inline comments to explain their workings in more detail.

analysis
- [test_metrics.py](analysis/test_metrics.py): Implementations for computing C-index, time-dependent Brier score, graphical calibration curves and survival function.

config:
- [run_randomsearch.sh](config/run_randomsearch.sh): Example configuration file to perform hyperparameter search on the Radboud University Data Science GPU cluster. More information about cluster can be found [here](https://wiki.cncz.science.ru.nl/Slurm).

eda_preprocessing
- [DataCreator.py](eda_preprocessing/DataCreator.py): Converts trajectories into sliding window and prediction horizon batches of desired size. 
- [DataPreprocessor.py](eda_preprocessing/DataPreprocessor.py): Loads and preprocesses the Tadpole data set as presented in [1].

hyperparameter_tuning
- [MatchNetHyperModel.py](hyperparameter_tuning/MatchNetHyperModel.py): Keras Tuner Model implementation of Match-Net for performing hyperparameter tuning.
- [MatchNetTuner.py](hyperparameter_tuning/MatchNetTuner.py): Keras Tuner derived implementation to tune Match-Net using the desired configuration options.
- [RandomSearchConfig.py](hyperparameter_tuning/RandomSearchConfig.py): Configuration file used to specify the selection ranges of parameters used for hyperparameter tuning.

model
- config
  - [MatchNetConfig.py](model/config/MatchNetConfig.py): Configuration file used to specify the hyperparameters of Match-Net for training.
- layers
  - [MCDropout.py](model/layers/MCDropout.py): Derived Tensorflow Dropout layer to implement Monte Carlo dropout. 
- [MatchNet.py](model/MatchNet.py): Implementation of Match-Net train- and teststep, loss function and AUROC and AUPRC metric computations.
- [model_builder.py](model/model_builder.py): Can be used to construct a Match-Net neural network, given the options specified in a [MatchNetConfig file](model/config/MatchNetConfig.py).

util/training
- [oversampler.py](util/training/oversampler.py): Random oversampling implementation for oversampling trajecties that contain an Alzheimer's disease event for a specified oversample ratio.

root
- [train_model.py](train_model.py): Script for training the Match-Net model, given a specified configuration.
- [tune_model.py](tune_model.py): Script for tuning the hyperparameters using a specified number of iterations of Random search.


## Questions
Please forward any questions about the usage of the software in this repository to : djessedirckx@gmail.com.