import torch
import os
from tqdm.auto import tqdm
import json, random
import numpy as np
import sys
import time

## getting the arguments
from opt_hier import config_parser
args = config_parser()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

from utils import *
from models.init_net.run import get_density_pnts
# from models.init_geo.run import get_density_pnts
from pyhocon import ConfigFactory
from models.masked_adam import MaskedAdam

from renderer import *
from models.apparatus import *
from preprocessing.recon_prior_hier import gen_geo
import datetime
from models.core.Strivec4d import Space_vec
from dataLoader.dan_video import DanDataset
import time

# generate the rays and the other necessarys....


#loading the models:
# we need to load all the models and interface one by one....,,,
ckpt = torch.load(args.ckpt, map_location=device)
kwargs = ckpt['kwargs']
kwargs.update({'device':device, "geo": geo, "local_dims":args.local_dims_final})
tensorf = eval(args.model_name)(**kwargs)
tensorf.load(ckpt)
#