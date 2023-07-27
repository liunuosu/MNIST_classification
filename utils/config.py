import yaml
from pathlib import Path


def get_config():
    with(open(Path("configs", "config_file.yaml"))) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config
