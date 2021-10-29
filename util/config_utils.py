from configparser import ConfigParser
from pathlib import Path

from model.config.MatchNetConfig import MatchNetConfig

def get_train_config(train_config_path: Path) -> MatchNetConfig:
    config_parser = ConfigParser()
    config_parser.read(train_config_path)

    # Read properties from config file and return model configuration
    return MatchNetConfig(
        pred_horizon=config_parser.getint('model_config', 'pred_horizon'),
        window_length=config_parser.getint('model_config', 'window_length'),
        cov_filters=config_parser.getint('model_config', 'cov_filters'),
        mask_filters=config_parser.getint('model_config', 'mask_filters'),
        cov_filter_size=config_parser.getint('model_config', 'cov_filter_size'),
        mask_filter_size=config_parser.getint('model_config', 'mask_filter_size'),
        cov_input_features=config_parser.getint('model_config', 'cov_input_features'),
        mask_input_features=config_parser.getint('model_config', 'mask_input_features'),
        dense_units=config_parser.getint('model_config', 'dense_units'),
        conv_blocks=config_parser.getint('model_config', 'conv_blocks'),
        dense_layers=config_parser.getint('model_config', 'dense_layers'),
        dropout_rate=config_parser.getfloat('model_config', 'dropout_rate'),
        learning_rate=config_parser.getfloat('train_config', 'learning_rate'),
        l1=config_parser.getfloat('model_config', 'l1'),
        l2=config_parser.getfloat('model_config', 'l2'),
        val_frequency=config_parser.getint('train_config', 'val_frequency'),
        mc_repeats=config_parser.getint('train_config', 'mc_repeats'),
        output_path=config_parser.get('train_config', 'output_path')
    )
