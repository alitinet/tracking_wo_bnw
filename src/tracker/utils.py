#########################################
# Still ugly file with helper functions #
#########################################

from .config import cfg, get_output_dir

import matplotlib.pyplot as plt
import numpy as np
import os
from os import path as osp
import cv2
import io
from PIL import Image
from scipy.optimize import linear_sum_assignment
from cycler import cycler as cy
from collections import defaultdict
from scipy.interpolate import interp1d

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# https://matplotlib.org/cycler/

# get all colors with
#colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)
colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
	"""
	Parameters
	----------
	boxes: (N, 4) ndarray or tensor or variable
	query_boxes: (K, 4) ndarray or tensor or variable
	Returns
	-------
	overlaps: (N, K) overlap between boxes and query_boxes
	"""
	if isinstance(boxes, np.ndarray):
		boxes = torch.from_numpy(boxes)
		query_boxes = torch.from_numpy(query_boxes)
		out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
	else:
		out_fn = lambda x: x

	box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
			(boxes[:, 3] - boxes[:, 1] + 1)
	query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
			(query_boxes[:, 3] - query_boxes[:, 1] + 1)

	iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
	ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
	ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
	overlaps = iw * ih / ua
	return out_fn(overlaps)

def plot_sequence(tracks, db, output_dir):
	"""Plots a whole sequence

	Args:
		tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
		db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
		output_dir (String): Directory where to save the resultind images
	"""

	print("[*] Plotting whole sequence to {}".format(output_dir))

	if not osp.exists(output_dir):
		os.makedirs(output_dir)

	# infinte color loop
	cyl = cy('ec', colors)
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	for i,v in enumerate(db):
		im_path = v['im_path']
		im_name = osp.basename(im_path)
		im_output = osp.join(output_dir, im_name)
		im = cv2.imread(im_path)
		im = im[:, :, (2, 1, 0)]

		fig, ax = plt.subplots(1,1)
		ax.imshow(im, aspect='equal')

		for j,t in tracks.items():
			if i in t.keys():
				t_i = t[i]
				ax.add_patch(
				plt.Rectangle((t_i[0], t_i[1]),
					  t_i[2] - t_i[0],
					  t_i[3] - t_i[1], fill=False,
					  linewidth=1.0, **styles[j])
				)

		plt.axis('off')
		plt.tight_layout()
		plt.draw()
		plt.savefig(im_output)
		plt.close()

def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
	#output_dir = get_output_dir("anchor_gt_demo")
	im_paths = blobs['im_paths']
	if not name:
		im0_name = osp.basename(im_paths[0])
	else:
		im0_name = str(name)+".jpg"
	im0 = cv2.imread(im_paths[0])
	im1 = cv2.imread(im_paths[1])
	im0 = im0[:, :, (2, 1, 0)]
	im1 = im1[:, :, (2, 1, 0)]

	im_scales = blobs['im_info'][0,2]

	tracks = tracks.data.cpu().numpy() / im_scales
	num_tracks = tracks.shape[0]

	#print(tracks.shape)
	#print(tracks)

	fig, ax = plt.subplots(1,2,figsize=(12, 6))

	ax[0].imshow(im0, aspect='equal')
	ax[1].imshow(im1, aspect='equal')

	# infinte color loop
	cyl = cy('ec', colors)
	loop_cy_iter = cyl()
	styles = defaultdict(lambda : next(loop_cy_iter))

	ax[0].set_title(('{} tracks').format(num_tracks), fontsize=14)

	for i,t in enumerate(tracks):
		t0 = t[0]
		t1 = t[1]
		ax[0].add_patch(
			plt.Rectangle((t0[0], t0[1]),
					  t0[2] - t0[0],
					  t0[3] - t0[1], fill=False,
					  linewidth=1.0, **styles[i])
			)
		ax[1].add_patch(
			plt.Rectangle((t1[0], t1[1]),
					  t1[2] - t1[0],
					  t1[3] - t1[1], fill=False,
					  linewidth=1.0, **styles[i])
			)

	if gt_tracks:
		for gt in gt_tracks:
			for i in range(2):
				ax[i].add_patch(
				plt.Rectangle((gt[i][0], gt[i][1]),
					  gt[i][2] - gt[i][0],
					  gt[i][3] - gt[i][1], fill=False,
					  edgecolor='blue', linewidth=1.0)
				)

	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	image = None
	if output_dir:
		im_output = osp.join(output_dir,im0_name)
		plt.savefig(im_output)
	else:
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
		image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	plt.close()
	return image

def interpolate(tracks):
	interpolated = {}
	for i, track in tracks.items():
		interpolated[i] = {}
		frames = []
		x0 = []
		y0 = []
		x1 = []
		y1 = []
		
		for f, bb in track.items():
			frames.append(f)
			x0.append(bb[0])
			y0.append(bb[1])
			x1.append(bb[2])
			y1.append(bb[3])

		if len(frames) > 1:
			x0_inter = interp1d(frames, x0)
			y0_inter = interp1d(frames, y0)
			x1_inter = interp1d(frames, x1)
			y1_inter = interp1d(frames, y1)
		
			for f in range(min(frames), max(frames)+1):
				bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
				interpolated[i][f] = bb
		else:
			interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

	return interpolated