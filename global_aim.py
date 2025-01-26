from aim import Run
from aim.pytorch_ignite import AimLogger

from omegaconf import DictConfig


# Global variable to store the Aim Run object
run = None


def init_aim(config: DictConfig) -> None:
    
    global run

    run = Run(experiment=config.experiment.name, log_system_params=True)
    run['config'] = config


def get_run():
    
    return run