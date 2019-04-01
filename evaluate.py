from network import GGCNN
from dataset import CornellDataset
from grasp import BoundingBoxes, detect_grasps

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from random import shuffle
import matplotlib.image as mpimg

DATA_PATH = 'dataset_190327_0029.h5'
MODEL_PATH = 'checkpoints/model_epoch_29.pth'
BATCH_SIZE = 16
NO_GRASPS = 1

VISUALISE_FAILURES = False
VISUALISE_SUCCESSES = False

def plot_output(rgb_img, depth_img, grasp_position_img, grasp_angle_img, ground_truth_bbs, no_grasps=1, grasp_width_img=None):
        """
        Visualise the outputs.
        """
        grasp_position_img = gaussian(grasp_position_img, 5.0, preserve_range=True)

        if grasp_width_img is not None:
            grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs)
        gs = detect_grasps(grasp_position_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, ang_threshold=0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(rgb_img)
        for g in gs:
            g.plot(ax)

        for g in gt_bbs:
            g.plot(ax, color='g')

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img)
        for g in gs:
            g.plot(ax, color='r')

        for g in gt_bbs:
            g.plot(ax, color='g')

        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(grasp_position_img, cmap='Reds', vmin=0, vmax=1)

        ax = fig.add_subplot(2, 2, 4)
        plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        plt.colorbar(plot)
        plt.show()

def calculate_iou_matches(grasp_positions_out, grasp_angles_out, ground_truth_bbs, no_grasps=1, grasp_width_out=None, min_iou=0.25):
    """
    Calculate a success score using the (by default) 25% IOU metric.
    Note that these results don't really reflect real-world performance.
    """
    succeeded = []
    failed = []
    for i in range(grasp_positions_out.shape[0]):
        grasp_position = grasp_positions_out[i, ].squeeze()
        grasp_angle = grasp_angles_out[i, :, :].squeeze()

        grasp_position = gaussian(grasp_position, 5.0, preserve_range=True)

        if grasp_width_out is not None:
            grasp_width = grasp_width_out[i, ].squeeze()
            grasp_width = gaussian(grasp_width, 1.0, preserve_range=True)
        else:
            grasp_width = None

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs[i, ].squeeze())
        gs = detect_grasps(grasp_position, grasp_angle, width_img=grasp_width, no_grasps=no_grasps, ang_threshold=0)
        for g in gs:
            if g.max_iou(gt_bbs) > min_iou:
                succeeded.append(i)
                break
        else:
            failed.append(i)

    return succeeded, failed

test_dataset = CornellDataset(data_path = DATA_PATH, train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_examples = len(test_dataset)
test_batches = len(test_dataloader)

ggcnn = GGCNN()
# cudnn.benchmark = True

ggcnn.cuda()

ggcnn.load_state_dict(torch.load(MODEL_PATH))

ggcnn.eval()

print("Evaluation examples: {}".format(test_examples))
print("Evaluation batches: {}".format(test_batches))
print("Start evaluating...")

total_right = 0
total_wrong = 0

for i,data in enumerate(test_dataloader, 0):
    rgb, depth, bbox = data
    depth = depth.float().cuda()

    pos_pred, cos_pred, sin_pred, width_pred = ggcnn(depth)

    width_pred = width_pred * 150.0

    pos_pred, cos_pred, sin_pred = pos_pred.cpu(), cos_pred.cpu(), sin_pred.cpu()

    width_pred = width_pred.cpu().detach().numpy()
    pos_pred = pos_pred.detach().numpy()

    depth = depth.cpu().detach().numpy()

    angle_pred = np.arctan2(sin_pred.detach().numpy(), cos_pred.detach().numpy())/2.0
    

    succeeded, failed = calculate_iou_matches(pos_pred, angle_pred, bbox, no_grasps=NO_GRASPS, grasp_width_out=width_pred)

    s = len(succeeded) * 1.0
    f = len(failed) * 1.0
    print('%s/%s\t%0.02f%s' % (s, s+f, s/(s+f)*100.0, '%'))
    
    total_right += s
    total_wrong += f

    
    if VISUALISE_FAILURES:
        print('Plotting Failures')
        shuffle(failed)
        for i in failed:
            plot_output(rgb[i, ], depth[i, ], pos_pred[i, ].squeeze(), angle_pred[i, ].squeeze(), bbox[i, ],
                        no_grasps=NO_GRASPS, grasp_width_img=width_pred[i, ].squeeze())

    if VISUALISE_SUCCESSES:
        print('Plotting Successes')
        shuffle(succeeded)
        for i in succeeded:
            plot_output(rgb[i, ], depth[i, ].squeeze(), pos_pred[i, ].squeeze(), angle_pred[i, ].squeeze(), bbox[i, ],
                        no_grasps=NO_GRASPS, grasp_width_img=width_pred[i, ].squeeze())

print('%s/%s\t%0.02f%s' % (total_right, total_right+total_wrong, total_right/(total_right+total_wrong)*100.0, '%'))
