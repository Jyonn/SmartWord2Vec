import os
import re

import yaml
from UniTok.classify import Classify


def get_config_value(config: Classify, path: str):
    path = path.split('.')
    value = config
    for key in path:
        value = value[key]
    return value


def format_config_path(config: Classify, path: str):
    dynamic_values = re.findall('{.*?}', path)
    for dynamic_value in dynamic_values:
        path = path.replace(dynamic_value, get_config_value(config, dynamic_value[1:-1]))
    return path


def init_config(data_path, model_path):
    data_config = yaml.safe_load(open(data_path))
    data_config = Classify(data_config)

    model_config = yaml.safe_load(open(model_path))
    model_config = Classify(model_config)

    model_config.model = model_config.model.upper()

    config = Classify(dict(model=model_config, data=data_config))
    data_config.data.dir = format_config_path(config, data_config.data.dir)
    model_config.save.path = format_config_path(config, model_config.save.path)

    model_config.save.log_path = os.path.join(model_config.save.path, 'log')
    os.makedirs(model_config.save.path, exist_ok=True)

    return config
