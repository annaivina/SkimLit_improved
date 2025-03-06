import os
import yaml


def load_config(config=""):
    if not config:
        raise ValueError("Config is not provided!")
    
    try:
        with open(config, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
