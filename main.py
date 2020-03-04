import argparse
import os
import imp
import datetime
import numpy as np
import experiments as exp

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, default='', help='config file with parameters of the experiment')
args_opt = parser.parse_args()

# sets unique directory name for writing all logs for experiment and saved models
log_directory = os.path.join('.', 'logs', args_opt.exp)

# loads experiment
exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
# loads parameter handler
ph = imp.load_source("",exp_config_file).config
ph['log_dir'] = log_directory # set log directory

experiment = getattr(exp, ph['experiment'])(ph)
experiment.run()