from omegaconf import DictConfig

config: DictConfig = None

def init_config(cfg):
    global config
    config = cfg

def get_config():
    return config