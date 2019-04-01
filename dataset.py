import torch.utils.data as data
import h5py
import numpy as np
import os
from glob import glob

class CornellDataset(data.Dataset):
    def __init__(self, data_path, train=True):
        
        self.train = train
        f = h5py.File(data_path, 'r')

        if self.train:
            self.depth_train = f['train/depth_inpainted']
            self.point_train = f['train/grasp_points_img']
            self.angle_train = f['train/angle_img']
            self.grasp_width_train = f['train/grasp_width']

        else:
            self.rgb_test = f['test/rgb']
            self.depth_test = f['test/depth_inpainted']
            self.point_test = f['test/grasp_points_img']
            self.angle_test = f['test/angle_img']
            self.grasp_width_test = f['test/grasp_width']
            self.gt_bbs = f['test/bounding_boxes']

    def __len__(self):
        if self.train:
            return self.depth_train.shape[0]
        else:
            return self.depth_test.shape[0]

    def __getitem__(self, index):
        if self.train:
            depth_data = np.expand_dims(np.array(self.depth_train[index, :]), 0)
            point_data = np.expand_dims(np.array(self.point_train[index, :]), 0)
            angle_data = np.array(self.angle_train[index, :])
            cos_data = np.expand_dims(np.cos(2*angle_data), 0)
            sin_data = np.expand_dims(np.sin(2*angle_data), 0)
            width_data = np.expand_dims(np.array(self.grasp_width_train[index, :]), 0)
            width_data = np.clip(width_data, 0, 150)/150.0

            return depth_data, point_data, cos_data, sin_data, width_data

        else:
            # depth_data = np.expand_dims(np.array(self.depth_test[index, :]), 0)
            # point_data = np.expand_dims(np.array(self.point_test[index, :]), 0)
            # angle_data = np.array(self.angle_test[index, :])
            # cos_data = np.expand_dims(np.cos(2*angle_data), 0)
            # sin_data = np.expand_dims(np.sin(2*angle_data), 0)
            # width_data = np.expand_dims(np.array(self.grasp_width_test[index, :]), 0)
            # width_data = np.clip(width_data, 0, 150)/150.0

            rgb_data = np.array(self.rgb_test[index, :])
            depth_data = np.expand_dims(np.array(self.depth_test[index, :]), 0)
            bbox_data = np.array(self.gt_bbs[index, :])

            return rgb_data, depth_data, bbox_data