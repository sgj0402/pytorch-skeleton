import hydra
from omegaconf import DictConfig, OmegaConf

import global_aim
import global_config

from model_runner import model_run


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig) -> None:

    global_aim.init_aim(config)
    global_config.init_config(config)

    model_run()
    
    
if __name__ == '__main__':
    main()