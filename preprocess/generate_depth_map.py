import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np
from imageio import imread, imwrite
import cv2
import pandas as pd


def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width))
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth

    depth_map = np.clip(depth_map, 0, 80)
    depth_map_unit16 = (depth_map*256.).astype(np.uint16)
    return depth_map_unit16


def get_mapping_index(sample_id):
    train_rand = pd.read_csv(
        os.path.join(
            root_dir, "kitti_3d", "mapping", "train_rand.txt"
        ),
        sep=',',
        header=None
    ).values.tolist()

    raw_index = train_rand[0][int(sample_id)]

    train_mapping = pd.read_csv(
        os.path.join(
            root_dir, "kitti_3d", "mapping", "train_mapping.txt"
        ),
        sep='\n',
        header=None
    ).values.tolist()

    raw_drive_name = train_mapping[raw_index - 1][0][11:-11]
    raw_frame_index = int(train_mapping[raw_index - 1][0][-10:])

    whole_pose = pd.read_csv(
        os.path.join(
            root_dir, "kitti_3d", "mapping", "pose", "{}".format(raw_drive_name), "pose.txt"
        ),
        sep=' ',
        header=None

    )
    return raw_drive_name, raw_frame_index


if __name__ == '__main__':
    root_dir = '/home/user/data/Dataset/'
    raw_dir = os.path.join(root_dir, "kitti_raw")
    split_dir = os.path.join(root_dir, "kitti_3d", "training")
    image_dir = os.path.join(split_dir, "image_2")
    calib_dir = os.path.join(split_dir, "calib")

    # depth_dir = os.path.join(split_dir, "depth_fut")
    depth_dir = os.path.join(split_dir, "depth_past")


    file_list = os.listdir(image_dir)

    for sample_id in file_list:
        sample_id = sample_id[:-4]

        calib_file = '{}/{}.txt'.format(calib_dir, sample_id)
        calib = kitti_util.Calibration(calib_file)

        raw_drive_name, raw_frame_id = get_mapping_index(sample_id)
        date = raw_drive_name[:10]

        lidar_dir = os.path.join(raw_dir, date, raw_drive_name, "velodyne_points", "data")
        future_image_dir = os.path.join(raw_dir, date, raw_drive_name, "image_02", "data")

        for time_step in range(3):

            # future_frame_id = raw_frame_id + time_step + 1
            #past
            future_frame_id = raw_frame_id - (time_step + 1)
            future_frame_id = str(future_frame_id).zfill(10)

            if not os.path.isfile(os.path.join(lidar_dir, "{}.bin".format(future_frame_id))):
                print(future_frame_id)
                continue


            # load point cloud
            lidar = np.fromfile(lidar_dir + '/' + '{}.bin'.format(future_frame_id), dtype=np.float32).reshape((-1, 4))[:, :3]
            image_file = '{}/{}.png'.format(future_image_dir, future_frame_id)
            image = imread(image_file)
            height, width = image.shape[:2]
            depth_map = generate_dispariy_from_velo(lidar, height, width, calib)
            imwrite(os.path.join(depth_dir, "{}_0{}.png".format(sample_id, str(time_step+1))), depth_map, format='png')
            # cv2.imwrite(depth_dir + '/' + sample_id + "_0" + str(time_step+1) + ".png", depth_map)

            # np.save(depth_dir + '/' + sample_id + "_0" + str(time_step+1), depth_map)
            print('Finish Depth Map {}_{}'.format(sample_id, str(time_step+1)))


