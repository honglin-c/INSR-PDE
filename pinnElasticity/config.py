import os
import argparse
import json
import shutil
from utils.file_utils import ensure_dirs


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase = 'train'):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.results_dir = os.path.join(self.exp_dir, 'results')

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # load saved config if not training
        if not self.is_train:
            assert os.path.exists(self.exp_dir)
            config_path = os.path.join(self.exp_dir, 'config.json')
            print(f"Load saved config from {config_path}")
            with open(config_path, 'r') as f:
                saved_args = json.load(f)
            for k, v in saved_args.items():
                if not hasattr(self, k):
                    self.__setattr__(k, v)
            return
        else:
            if os.path.exists(self.exp_dir):
                response = input('Experiment log/model already exists, overwrite? (y/n) ')
                if response != 'y':
                    exit()
                shutil.rmtree(self.exp_dir, ignore_errors=True)
            ensure_dirs([self.log_dir, self.model_dir, self.results_dir])

        # save this configuration for backup
        backup_dir = os.path.join(self.exp_dir, "backup")
        ensure_dirs(backup_dir)
        os.system(f"cp *.py {backup_dir}/")
        # os.system(f"mkdir {backup_dir}/models | cp models/*.py {backup_dir}/models/")
        # os.system(f"mkdir {backup_dir}/utils | cp utils/*.py {backup_dir}/utils/")
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        if self.is_train:
            # model configuration
            self._add_network_config_(parser)

            # training or testing configuration
            self._add_training_config_(parser)
        else:
            self._add_testing_config_(parser)

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="checkpoints", 
            help="path to project folder where models and logs will be saved")
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument('--network', type=str, default='siren', choices=['siren', 'grid'])
        group.add_argument('--num_hidden_layers', type=int, default=3)
        group.add_argument('--hidden_features', type=int, default=256)
        group.add_argument('--nonlinearity',type=str, default='sine')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--ckpt', type=str, default=None, required=False, help="checkpoint at x timestep to restore")
        group.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        group.add_argument('--max_n_iters', type=int, default=50000, help='number of epochs to train per scale')
        group.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0005')
        group.add_argument('--grad_clip', type=float, default=0.2, help='grad clipping, l2 norm')
        group.add_argument('--early_stop', action='store_true', help="early_stopping")
        
        group.add_argument('--dt', type=float, default=0.1)
        group.add_argument('-T','--n_timesteps', type=int, default=1)
        group.add_argument('-sr', '--sample_resolution', type=int, default=128)
        group.add_argument('-vr', '--vis_resolution', type=int, default=32)
        group.add_argument('--fps', type=int, default=10)

        group.add_argument('--boundary_cond', type=str, default='zero', choices=['zero', 'none'])
        group.add_argument('--sample', type=str, default='random', choices=['random', 'uniform', 'random+uniform', 'fixed'],
                            help='The sampling strategy to be used during the training.')

        group.add_argument('--t_range', type=float, default=1.0, help='time range')

        group.add_argument('--density', type=float, default=1.0, help='density')
        group.add_argument('--ratio_arap', type=float, default=1e0, help='ratio for ARAP energy')
        group.add_argument('--ratio_volume', type=float, default=1e1, help='ratio for volume-preserving energy')
        group.add_argument('--gravity_g', type=float, default=-9.8, help='gravity acceleration')

        group.add_argument('--lambda_main', type=float, default=1.0, help='coeffcients for lambda')

        group.add_argument('--enable_collision', type=bool, default=False, help='enable collsion or not')
        group.add_argument('--ratio_collision', type=float, default=1e6, help='time range')
        group.add_argument('--plane_height', type=float, default=-2.0, help='time range')
    
    def _add_testing_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        group.add_argument('-vr', '--vis_resolution', type=int, default=32)
        group.add_argument('--fps', type=int, default=10)
