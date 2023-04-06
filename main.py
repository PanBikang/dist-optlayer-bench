
import os
import sys
import math
import copy
import json
import yaml
import shutil
import argparse
import setproctitle

from dist_train import FedDistManager, DecenDistManager, BilevelFedDistManager

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # default_config_path = os.path.join("old_config", 'mnist_config', "lenet-mnist")
    parser.add_argument('--config', type=str, default='bil-fed-fc-cifar10-adam')
    parser.add_argument("--save_dir", type=str, default='lenet_basic_test')
    args = parser.parse_args()
    
    config_path = os.path.join('config', args.config + '.yaml')
    
    # load all yaml config dictionary settings
    with open(config_path, 'r') as f:
        config_dict = yaml.full_load(f)
    slurm_id = os.getenv("SLURM_JOB_ID")
    # if this job works in slurm cluster, then take slurm id as filefold name
    if slurm_id:
        t = slurm_id
    # else take default name
    elif config_dict['save'] is None:
        t = '{}.{}'.format(config_dict['dataset'], config_dict['model_name'])
        if config_dict['model_name'] == 'lenet':
            t += '.nHidden:{}.proj:{}'.format(config_dict['nHidden'], 
                                              config_dict['proj'])
        elif config_dict['model_name'] == 'fc':
            t += '.nHidden:{}'.format(config_dict['nHidden'])
            if config_dict['bn']:
                t += '.bn'
        elif config_dict['model_name'] == 'optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(config_dict['nHidden'],
                                                      config_dict['nineq'], 
                                                      config_dict['eps'])
            if config_dict['bn']:
                t += '.bn'
        elif config_dict['model_name'] == 'optnet-eq':
            t += '.nHidden:{}.neq:{}'.format(config_dict['nHidden'], 
                                             config_dict['neq'])
        elif config_dict['model_name'] == 'lenet-optnet':
            t += '.nHidden:{}.nineq:{}.eps:{}'.format(config_dict['nHidden'], 
                                                      config_dict['nineq'], 
                                                      config_dict['eps'])
    setproctitle.setproctitle('bamos.'+t)
    config_dict['save'] = os.path.join("experiment", args.save_dir, t)
    if not os.path.exists(os.path.join("experiment", args.save_dir)):
        os.makedirs(os.path.join("experiment", args.save_dir), exist_ok=True)
    
    if os.path.exists(config_dict['save']):
        shutil.rmtree(config_dict['save'])
    os.makedirs(config_dict['save'], exist_ok=True)
    try:
        shutil.copy(config_path, config_dict['save'])
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())
    new_config_dict = {'config_dict': config_dict}
    exp_env = BilevelFedDistManager(config_dict['save'])
    exp_env.run_exp(config_dict)
    