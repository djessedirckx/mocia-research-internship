from model_builder import build_model
from MatchNetConfig import MatchNetConfig

config_test = MatchNetConfig(10, 10, 10, 10, (10,10), (10, 10), 10, 5, 0.001, 16)
model_test = build_model(config_test)