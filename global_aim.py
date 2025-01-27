from aim import Run

from omegaconf import DictConfig


# Global variable to store the Aim Run object
run = None


def init_aim(config: DictConfig) -> None:

    if config.setting.mode == 'train':

        if config.setting.train.is_resume and config.setting.aim.run_hash is None:
            raise ValueError("Resume is enabled but run_hash is not provided.")
        
        if config.setting.train.is_resume is False and config.setting.aim.run_hash is not None:
            print("Resume is disabled but run_hash is provided. Ignoring run_hash.")
            config.setting.aim.run_hash = None
    
    global run

    run = Run(run_hash=config.setting.aim.run_hash,
              experiment=config.experiment.name,
              log_system_params=True)
    
    run['config'] = config


def get_run():
    
    return run