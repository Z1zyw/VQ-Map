import yaml
import time
from argparse import ArgumentParser

class BasicConfig:
    def __init__(self, cfg=None):
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class EvaluateConfig(BasicConfig):
    def __init__(self, cfg=None):
        # evaluate
        super().__init__(cfg)


class TrainConfig(BasicConfig):
    def __init__(self, cfg=None):
        # train
        self.exp_name: str = "Train_exp_" + str(round(time.time()))
        self.model_cfg: str = ""
        self.data_cfg: str = ""
        self.num_epochs: int = 1
        self.save_interval: int = 1
        self.val_interval: int = 1
        self.accumulation_steps: int = 1
        self.seed = 1231
        # self.log_dir: str = 'logs'
        self.ckpt_dir: str = 'bin'
        self.log_interval: int = 10
                
        # dataloader
        self.batch_size: int = 2
        self.num_workers: int = 1
        self.shuffle: bool = True
        self.pin_memory: bool = True
        self.drop_last: bool = False
        self.prefetch_factor: int = 2
        
        # resume
        self.resume: bool = False
        self.resume_dir: str = ''
        self.resume_epoch: int = 0
        
        # optimizer
        self.optimizer: str = 'adam'
        self.lr: float = 1e-3
        self.lr_backbone: float = 1e-4
        self.weight_decay: float = 1e-4
        self.momentum: float = 0.9
        
        # lr scheduler
        self.lr_scheduler: str = 'step'
        self.lr_step_size: int = 10
        self.lr_gamma: float = 0.1
        
        # other 
        self.debug: bool = True
        self.test: bool = False
        self.use_new_optimizer: bool = False
        self.resume_path_other: str = ''
        super().__init__(cfg)


def load_config(config_file, config: BasicConfig=None):
    # if BasicConfig is not None, update the config, else return the config
    cfg = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        
    if cfg is None and config is None:
        raise ValueError("Config is None and BasicConfig is None(no default config), please check!")
    
    if config is not None:
        if cfg is None:
            save_config(config, config_file)
        else:
            config.update(cfg)
        return config

    return cfg


def save_config(config, config_file):
    with open(config_file, 'w') as f:
        yaml.dump(config.__dict__, f)
        

def parser2config(parser: ArgumentParser , config: BasicConfig, verbose:bool=True):
    # write parser args to config
    parser_args = vars(parser.parse_args())
    for k, v in parser_args.items():
        if getattr(config, k) is not None and v is not None:
            setattr(config, k, v)
    if verbose:
        print("\n ==> Config:")
        for k, v in parser_args.items():
            print(f" {k}: {v}")
    return config
        
