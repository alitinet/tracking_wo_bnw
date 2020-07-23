from sacred import Experiment
import os.path as osp
import os
import numpy as np
import yaml
import cv2

import torch
from torch.utils.data import DataLoader

from tracktor.config import get_output_dir, get_tb_dir
from tracktor.reid.solver import Solver
from tracktor.datasets.factory import Datasets
from tracktor.reid.resnet import resnet50, resnet34, resnet18

ex = Experiment()
ex.add_config('experiments/cfgs/reid.yaml')

Solver = ex.capture(Solver, prefix='reid.solver')

@ex.automain
def my_main(_config, reid):
    # set all seeds
    torch.manual_seed(reid['seed'])
    torch.cuda.manual_seed(reid['seed'])
    np.random.seed(reid['seed'])
    torch.backends.cudnn.deterministic = True

    print(_config)

    output_dir = osp.join(get_output_dir(reid['module_name']), reid['name'])
    tb_dir = osp.join(get_tb_dir(reid['module_name']), reid['name'])

    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    #########################
    # Initialize dataloader #
    #########################
    print("[*] Initializing Dataloader")

    db_train = Datasets(reid['db_train'], reid['dataloader'])
    db_train = DataLoader(db_train, batch_size=1, shuffle=True)

    if reid['db_val']:
        db_val = Datasets(reid['db_val'], reid['dataloader'])
        db_val = DataLoader(db_val, batch_size=1, shuffle=True)
        #db_val = None
    else:
        db_val = None

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    network = resnet50(pretrained=True, **reid['cnn'])
    # network.load_state_dict(torch.load("output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth",
                                 map_location=lambda storage, loc: storage))

    
    network.train()
    network.cuda()

    ##################
    # Begin training #
    ##################
    print("[*] Solving ...")

    # build scheduling like in "In Defense of the Triplet Loss for Person Re-Identification"
    # from Hermans et al.
    lr = reid['solver']['optim_args']['lr']
    iters_per_epoch = len(db_train)
    
    # we want to keep lr until iter 3000 and from there to iter 5000 a exponential decay
    l = eval("lambda epoch: 1 if epoch*{} < 3000 else 0.001**((epoch*{} - 3000)/(5000-3000))".format(
                                                                iters_per_epoch,  iters_per_epoch))

    max_epochs = 5000 // len(db_train.dataset) + 1 if 5000 % len(db_train.dataset) else 5000 // len(db_train.dataset)
    solver = Solver(output_dir, tb_dir, lr_scheduler_lambda=l)
    solver.train(network, db_train, db_val, max_epochs, 50, model_args=reid['model_args'])




