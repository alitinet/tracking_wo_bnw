import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.maskrcnn_fpn import MaskRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
import matplotlib.pyplot as plt

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


def plot(img, boxes, masks, prob_threshold=0.5):
    fig, ax = plt.subplots(1, dpi=96)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    width, height,_ = img.shape

    ax.imshow(img)
    fig.set_size_inches(width / 80, height / 80)

    colors = np.random.rand(len(boxes), 3)

    # plot bounding box and mask
    for i, box in enumerate(boxes):

        # add bounding box
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color=colors[i],
            linewidth=1.0)
        ax.add_patch(rect)

        mask = masks[i]

        # add mask
        for channel in range(3):
            img[:, :, channel] = np.where(mask >= prob_threshold,
                                          img[:, :, channel] * 0.3 + 0.7 * colors[i][channel] * 255,
                                          img[:, :, channel])

    ax.imshow(img)
    plt.axis('off')
    plt.savefig('tracktors.png')



@ex.automain
def main(tracktor, reid, _config, _log, _run):


    sacred.commands.print_config(_run)

    # set all seeds
    #torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    obj_detect = MaskRCNN_FPN(num_classes=2)

    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = Datasets(tracktor['dataset'])
    for seq in dataset:

        tracker.reset()

        start = time.time()

        _log.info(f"Tracking: {seq}")

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)

        frames = []
        for j in range(35):
            frames.append(next(iter(data_loader)))

        for i, frame in enumerate(tqdm(data_loader)):
            if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                with torch.no_grad():
                    tracker.step(frame)
                num_frames += 1
        results = tracker.get_results()

        # frame_result = results[0][0]
        # plot(frames[0]['img'][0], [frame_result[0]], [frame_result[1]])

        time_total += time.time() - start

        _log.info(f"Tracks found: {len(results)}")
        _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

        #mot_results = {}
        #for track in results:
        #    for frame in results[track]:
        #        mot_results[track][frame] = np.concatenate(results[track][frame][0], results[track][frame][2])


        #if seq.no_gt:
        #    _log.info(f"No GT data for evaluation available.")
        #else:
        #    mot_accums.append(get_mot_accum(mot_results, seq))

        _log.info(f"Writing predictions to: {output_dir}")
        seq.write_results(results, output_dir)