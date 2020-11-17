import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

'''
'''
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pcdet.datasets.mapping import mapping
import torch
from torch import nn
import cv2
from pcdet.datasets.simplevis import nuscene_vis
import matplotlib.pyplot as plt
import seaborn as sns

'''
'''


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1,
                                                                                         4)  # kitti:(-1,4), nuscenes:(-1,5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger)
    # dict_keys(['points', 'frame_id', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points'])
    # demo_dataset = NuScenesDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #                                root_path=None, logger=logger)  # change demo_dataset
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    # ---------------------------
    # Kitti
    # ---------------------------
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            # ---------------------------
            # Visualization
            # ---------------------------
            logger.info(f'Visualized sample index: \t{idx + 1}')
            points = data_dict['points']
            pc_range = np.array([0, -39.68, -3, 69.12, 39.68, 1])
            voxel_size = np.array([0.16, 0.16, 4])
            sensor_origins = np.array([0.0, 0.0, 0.0])
            logodds = mapping.compute_logodds_no_timestamp(points, sensor_origins, pc_range, min(voxel_size))
            filter = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
                               groups=1,
                               bias=True)
            occupancy = torch.sigmoid(torch.from_numpy(logodds))  # (occupied:0.7,unknown:0.5,free:0.4)
            occupancy = occupancy.reshape((-1, 25, 496, 432))
            # print(occupancy.shape)#torch.Size([1, 40, 512, 512])
            visibility = filter(occupancy)
            visibility = np.squeeze(visibility.detach().numpy())
            # ---------------------------
            # show n.10 stage visibility features
            # ---------------------------
            plt.figure('n.10 stage')
            sns.heatmap(occupancy[0, 10, :, :], cmap='rainbow')
            plt.show()

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)
    # ---------------------------
    # NuScenes
    # ---------------------------
    # with torch.no_grad():
    #     for idx, data_dict in enumerate(demo_dataset):
    #         # ---------------------------
    #         # Visualization
    #         # ---------------------------
    #         if idx + 1 in [10]:
    #             logger.info(f'Visualized sample index: \t{idx + 1}')
    #             # print(data_dict.keys())#dict_keys(['points', 'frame_id', 'metadata', 'time_stamps', 'origins',
    #             # 'cam_front_path', 'gt_boxes', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points'])
    #             points = data_dict['points']
    #             gt_boxes = data_dict['gt_boxes']
    #             pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    #             voxel_size = np.array([0.2, 0.2, 8.0])
    #             time_stamps = data_dict['time_stamps']
    #             original_points = np.delete(points, 3, axis=1)
    #             if time_stamps is not None:
    #                 sensor_origins = data_dict['origins']
    #                 logodds = mapping.compute_logodds(original_points, sensor_origins, time_stamps, pc_range,
    #                                                   min(voxel_size))  # with timestamps
    #             else:
    #                 sensor_origins = np.array([0.0, 0.0, 0.0])
    #                 logodds = mapping.compute_logodds_no_timestamp(original_points, sensor_origins, pc_range,
    #                                                                min(voxel_size))  # no timestamps
    #
    #             filter = nn.Conv2d(in_channels=40, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
    #                                groups=1,
    #                                bias=True)
    #             occupancy = torch.sigmoid(torch.from_numpy(logodds))  # (occupied:0.7,unknown:0.5,free:0.4)
    #             occupancy = occupancy.reshape((-1, 40, 512, 512))
    #             # print(occupancy.shape)#torch.Size([1, 40, 512, 512])
    #             visibility = filter(occupancy)
    #             visibility = np.squeeze(visibility.detach().numpy())
                # ---------------------------
                # show 40 stages visibility features
                # ---------------------------
                # plt.figure('40 stages')
                # for i in range(occupancy.shape[1]):
                #     plt.subplot(5, 8, i + 1)
                #     sns.heatmap(occupancy[0, i, :, :], cmap='rainbow')
                # plt.axis('off')
                # plt.show()
                # ---------------------------
                # show n.20 stage visibility features
                # ---------------------------
                # plt.figure('n.20 stage')
                # sns.heatmap(occupancy[0, 20, :, :], cmap='rainbow')
                # plt.show()
                # ---------------------------
                # show one stage learned visibility feature
                # ---------------------------
                # plt.figure('learned visibility')
                # sns.heatmap(visibility, cmap='rainbow')
                # plt.show()
                # ---------------------------
                # show original image
                # ---------------------------
                # cam_path = Path('/mrtstorage/datasets_public/nuscenes/full_v1/mini/samples/CAM_FRONT') / data_dict[
                #     'cam_front_path']
                # print(data_dict['cam_front_path'])
                # lena = mpimg.imread(str(cam_path) + '.jpg')
                # lena.shape
                # plt.imshow(lena)
                # plt.axis('off')
                # plt.show()
                # ---------------------------
                # show original bev map
                # ---------------------------
                # bev_map = nuscene_vis(points)
                # plt.figure('bev map' + str(idx + 1))
                # plt.imshow(bev_map, cmap='Blues')
                # plt.axis('off')
                # plt.show()
                # ---------------------------
                # show original points with rays
                # ---------------------------
                # fig = V.visualize_pts(points)
                # pc_vis_idx = np.random.choice(range(points.shape[0]), 5000)
                # for i in pc_vis_idx:
                #     mlab.plot3d([0, points[i][0]], [0, points[i][1]], [0, points[i][2]], color=(0.1, 1, 1),
                #                 line_width=0.2,
                #                 tube_radius=None, figure=fig)
                # data_dict.pop('cam_front_path')
                # data_dict = demo_dataset.collate_batch([data_dict])
                # # print(data_dict.keys())#dict_keys(['points', 'use_lead_xyz', 'voxels', 'voxel_coords',
                # # 'voxel_num_points', 'batch_size'])
                # load_data_to_gpu(data_dict)
                # pred_dicts, _ = model.forward(data_dict)
                #
                # V.draw_scenes(
                #     points=points, gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
