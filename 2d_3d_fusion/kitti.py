import glob
import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import mlab

import utils
from From2dTo3d_tradition import uvToXYZ


# 左/右相机内参数、旋转、平移矩阵
# 左/右相机内参数、旋转、平移矩阵
leftIntrinsic = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
                          [0.000000e+00, 9.569251e+02, 2.241806e+02],
                          [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# leftIntrinsic = np.array([[9.842439e+02,0.000000e+00,6.900000e+02],
#                           [0.000000e+00,9.808141e+02,2.331966e+02],
#                           [0.000000e+00,0.000000e+00,1.000000e+00]])

leftRotation = np.array([[0.9999758, -0.005267463, -0.004552439],
                         [0.005251945, 0.9999804, -0.003413835],
                         [0.004570332, 0.003389843, 0.9999838]])

# leftRotation = np.array([[1, 0, 0],
#                          [0,1,0],
#                          [0,0,1]])

leftTranslation = np.array([[5.956621e-02],
                            [2.900141e-04],
                            [2.577209e-03]])

# leftTranslation = np.array([[2.573699e-16],
#                             [-1.059758e-16],
#                             [1.614870e-16]])

rightIntrinsic = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                           [0.000000e+00, 9.019653e+02, 2.242509e+02],
                           [0.000000e+00, 0.000000e+00, 1.000000e+00]])

rightRotation = np.array([[9.995599e-01, 1.699522e-02, -2.431313e-02],
                          [-1.704422e-02, 9.998531e-01, -1.809756e-03],
                          [2.427880e-02, 2.223358e-03, 9.997028e-01]])

rightTranslation = np.array([[-4.731050e-01],
                             [5.551470e-03],
                             [-5.250882e-03]])

R2_rect = np.array([[9.998817e-01, 1.511453e-02, -2.841595e-03],
                    [-1.511724e-02, 9.998853e-01, -9.338510e-04],
                    [2.827154e-03, 9.766976e-04, 9.999955e-01]])

R0_rect = np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03],
                    [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03],
                    [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]])

R3_rect = np.array([[9.998321e-01,-7.193136e-03,1.685599e-02],
                    [7.232804e-03,9.999712e-01,-2.293585e-03],
                    [-1.683901e-02,2.415116e-03,9.998553e-01]])
def get_all_index_in_list(L, item):
    """
    get all the indexies of the same items in the list
    :param L: list
    :param item: item to be found
    :return: the indexies of all same items in the list
    """

    return [index for (index, value) in enumerate(L) if value == item]

class KITTI(object):
    """
    Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,
    Andreas Geiger, Philip Lenz, and Raquel Urtasun,
    CVPR, 2012.
    """
    def __init__(self, dataset_path):
        '''
        :param dataset_path: path to the KITTI dataset
        '''
        super(KITTI, self).__init__()
        self.dataset_path = dataset_path
        self.get_sequence_list()
        self.categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        # self.colors = sns.color_palette(palette='muted', n_colors=len(self.categories))

    def get_sequence_list(self):
        """
        :return: the sequence list
        """

        # used to store the sequence info
        self.sequence_list = []

        # get all video names
        vid_names = os.listdir('{}/velodyne'.format(self.dataset_path))
        print(vid_names)
        vid_names.sort()
        self.sequence_num = len(vid_names)

        for vid in vid_names:
            # store information of a sequence
            sequence = dict()
            vid=vid.split('.')[0]

            sequence['name'] = vid
            sequence['img_list'] = glob.glob('{}/image_2/{}.png'.format(self.dataset_path, vid))
            sequence['img_list'].sort()
            sequence['img_size'] = self.get_sequence_img_size(sequence['img_list'][0])
            sequence['pcloud_list'] = glob.glob('{}/velodyne/{}.bin'.format(self.dataset_path, vid))
            sequence['pcloud_list'].sort()
            sequence['label_list'] = self.get_sequence_labels(vid)   # get labels in this sequence
            sequence['calib'] = self.get_sequence_calib(vid)

            self.sequence_list.append(sequence)

    def get_sequence_img_size(self, initial_img_path):
        """
        get the size of image in the sequence
        :return: image size
        """

        img = cv2.imread(initial_img_path)  # read image

        img_size = dict()

        img_size['height'] = img.shape[0]
        img_size['width'] = img.shape[1]

        return img_size

    def get_sequence_calib(self, sequence_name):
        """
        get the calib parameters
        :param sequence_name: sequence name
        :return: calib
        """

        # load data
        sequence_calib_path = '{}/calib/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_calib_path, 'r') as f:
            calib_lines = f.readlines()

        calib = dict()
        calib['P0'] = np.array(calib_lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P1'] = np.array(calib_lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P2'] = np.array(calib_lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['P3'] = np.array(calib_lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Rect'] = np.array(calib_lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
        calib['Tr_velo_cam'] = np.array(calib_lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        calib['Tr_imu_velo'] = np.array(calib_lines[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

        return calib

    def get_sequence_labels(self, sequence_name):
        """
        get labels for all frames in the sequence
        :param sequence_name: sequence name
        :return: the labels of a sequence
        """

        sequence_label_path = '{}/label_2/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_label_path, 'r') as f:
            labels = f.readlines()

        # parse each line
        # 1 frame number, 2 track id, 3 object type, 4 truncated, 5 occluded (0: full visible, 1: partly occluded, 2: largely occluded),
        # 6 alpha, 7-10 2d bbox in RGB image, 11-13 dimension (height, width, length in meters), 14-16 center location (x, y, z in meters),
        # 17 rotation around Y-axis
        frame_id_list = []
        object_list = []
        for line in labels:
            # process each line
            line = line.split()
            object_type, truncat, occ, alpha, l, t, r, b, height, width, lenght, x, y, z, rotation = line
            frame_id=0

            # map string to int or float
            alpha, l, t, r, b, height, width, lenght, x, y, z, rotation = map(float,
                                                                              [alpha, l, t, r, b, height, width, lenght,
                                                                               x, y, z, rotation])

            if object_type != 'DontCare':
                object = dict()  # store the information of this object
                object['object_type'] = object_type
                object['truncat'] = truncat
                object['occ'] = occ
                object['alpha'] = alpha
                object['bbox'] = [l, t, r, b]
                object['dimension'] = [height, width, lenght]
                object['location'] = [x, y, z]
                object['rotation'] = rotation

                object_list.append(object)
                frame_id_list.append(frame_id)

        # number of frames in this sequence
        frame_num = frame_id + 1

        # collect labels for each single frame
        sequence_label = []  # the labels of all frames in the sequence
        for i in range(frame_num):
            # get all the labels in frame i
            frame_ids = get_all_index_in_list(frame_id_list, i)
            if len(frame_ids) > 0:
                frame_label = object_list[frame_ids[0]:frame_ids[-1] + 1]
                sequence_label.append(frame_label)
            else:
                # for some frames, there are no objects
                sequence_label.append([])

        return sequence_label

    def show_sequence_pointcloud(self, vid_id, img_region=False, vis_box=True, save_img=False, save_path=None):
        """
        visualize the sequence in point cloud
        :param vid_id: id of the sequence, starting from 0
        :param img_region: only show point clouds in RGB image
        :param vis_box: show 3D boxes or not
        :return: none
        """

        assert 0 <= vid_id < len(self.sequence_list), 'The sequence id should be in [0, {}]'.format(
            str(self.sequence_num - 1))
        sequence = self.sequence_list[vid_id]
        sequence_name = sequence['name']
        pcloud_list = sequence['pcloud_list']
        labels = sequence['label_list']
        img_size = sequence['img_size']
        calib = sequence['calib']

        # create folder to save image if not existing
        if save_img:
            if save_path is None:
                if vis_box:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name + '_3D_box')
                else:
                    save_path = os.path.join('./seq_pointcloud_vis', sequence_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # # load point cloud
        # pcloud = np.fromfile(pcloud_list[0], dtype=np.float32).reshape(-1, 4)
        #
        # pcloud_fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
        # plt = mlab.points3d(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], mode='point', figure=pcloud_fig)
        # # another way is to use animate function in mlab to play cloud
        # # but somehow, it sometimes works, but sometimes fails
        #
        # @mlab.animate(delay=100)
        # def anim():
        #     for i in range(1, len(pcloud_list)):
        #         pcloud_name = pcloud_list[i]
        #         print(pcloud_name)
        #         # load point cloud
        #         pcloud = np.fromfile(pcloud_name, dtype=np.float32).reshape(-1, 4)
        #         plt.mlab_source.reset(x=pcloud[:, 0], y=pcloud[:, 1], z=pcloud[:, 2])
        #         mlab.savefig(filename='temp_img2/' + str(i) + '.png')
        #         yield
        #
        # anim()
        # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=50.0)
        # mlab.show()

        # visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='32line Lidar', width=1080, height=720, left=300, top=150, visible=True)
        point_cloud = o3d.geometry.PointCloud()
        pointcloud = np.fromfile(pcloud_list, np.float32).reshape(-1, 4)
        point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3].reshape(-1, 3))
        vis.add_geometry(point_cloud)
        calib = utils.get_sequence_calib()

        # load and show 3d boxes
        if vis_box:
            for object in labels:
                object_type = object['object_type']
                corners_3d_cam2 = utils.transform_3dbox_to_pointcloud(object['dimension'], object['location'],
                                                                      object['rotation'], calib,'P2')
                corners_3d_cam3 = utils.transform_3dbox_to_pointcloud(object['dimension'], object['location'],
                                                                      object['rotation'], calib, 'P3')
                x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
                corners_3d_velo = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
                # corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
                lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                      [0, 4], [1, 5], [2, 6], [3, 7]])

                # lines_box = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
                # 设置点与点之间线段的颜色
                colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
                # 创建Bbox候选框对象
                line_set = o3d.geometry.LineSet()
                # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
                line_set.lines = o3d.utility.Vector2iVector(lines_box)
                # 设置每条线段的颜色
                line_set.colors = o3d.utility.Vector3dVector(colors)
                # 把八个顶点的空间信息转换成o3d可以使用的数据类型
                line_set.points = o3d.utility.Vector3dVector(corners_3d_velo)
                # 将矩形框加入到窗口中
                vis.add_geometry(line_set)

            vis.get_render_option().point_size = 1  # 点云大小
            vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
        vis.run()

        pcloud_name=pcloud_list[0]
        if save_img:
            mlab.savefig(filename=os.path.join(save_path, pcloud_name.split('/')[-1].split('.')[0] + '.png'))
        else:
            mlab.savefig(
                filename='temp_img.png')  # save the visualization image (this line is necessary for visualization)

        # mlab.show()   # do NOT use this line, as it will get the focus and pause the code
        mlab.close(all=True)
        if not save_img:
            os.remove(path='temp_img.png')  # remove temp image file

if __name__ == "__main__":
    kitti = KITTI('data/kitti/training')
    kitti.show_sequence_pointcloud('000045')
