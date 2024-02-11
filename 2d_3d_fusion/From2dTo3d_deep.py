import numpy as np
import utils
import open3d as o3d
from skimage.transform import resize


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''

    inv_Tr = np.zeros_like(Tr)  # 3x4

    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])

    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])

    return inv_Tr


def visualize(sequence):
    sequence_label_path = 'data/kitti/training/label_2/{}.txt'.format(sequence)
    with open(sequence_label_path, 'r') as f:
        labels = f.readlines()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='prediction', width=1080, height=720, left=300, top=150, visible=True)
    pointcloud = np.fromfile("data/kitti/training/velodyne/{}.bin".format(sequence), np.float32).reshape(-1, 4)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3].reshape(-1, 3))
    vis.add_geometry(point_cloud)
    calib = utils.get_sequence_calib(sequence)
    loadData = np.load('kitty_open3d/{}_disp.npy'.format(sequence))
    loadData = np.asarray(resize(loadData.squeeze(), [375, 1242]))

    # get bbox according to txt label
    for line in labels:
        Point_velodyne = []
        line = line.split()
        object_type, truncat, occ, alpha, lx, ly, rx, ry, height, width, lenght, x, y, z, rotation = line

        lx = float(lx)
        ly = float(ly)
        rx = float(rx)
        ry = float(ry)
        height = float(height)
        width = float(width)
        lenght = float(lenght)
        x = float(x)
        y = float(y)
        z = float(z)
        rotation = float(rotation)
        if object_type != 'DontCare':
            calib = utils.get_sequence_calib(sequence)
            # get image 3D bounding box
            corners_3d_img = utils.transform_3dbox_to_image([height, width, lenght], [x, y, z], rotation, calib, 'P2')
            corners_3d_img = corners_3d_img.astype(int)
            corordinate_cam2 = []
            # transform 8 bbox points to cam corordinate
            # for i in range(len(corners_3d_img)):
            #     lx=int(corners_3d_img[i][0])
            #     ly=int(corners_3d_img[i][1])
            #     depth=721 * 0.54 / loadData[ly][lx]/1000
            #     point_3d=[lx*depth,ly*depth,depth]
            #     points_cam2_3d.append(point_3d)

            # only get center point
            center_2d = [(corners_3d_img[2][0] + corners_3d_img[7][0]) / 2,
                         (corners_3d_img[2][1] + corners_3d_img[7][1]) / 2]
            lx = int(center_2d[0])
            ly = int(center_2d[1])
            depth = 721 * 0.54 / loadData[ly][lx] / 1242
            print(depth)
            corordinate_cam2.append([lx * depth, ly * depth, depth])
            print(corordinate_cam2)
            corordinate_cam0 = []
            for i in range(len(corordinate_cam2)):
                point_3d = utils.camera_2_to_camera_0(corordinate_cam2[i], calib)
                corordinate_cam0.append(point_3d)
                corordinate_cam0 = np.asarray(corordinate_cam0)
            # point_3d_cam0 = np.asarray(utils.camera_2_to_camera_0(point_3d_cam2, calib))

            # 从cam0坐标转换到velo坐标系 .T转置操作
            points_cam0_3d = np.asarray(points_cam0_3d)
            calib = utils.Calibration('data/kitti', from_video=True)
            points_3d_velodyne = calib.project_rect_to_velo(points_cam0_3d)

            # draw box in velodyne
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
            line_set.points = o3d.utility.Vector3dVector(points_3d_velodyne)
            # 将矩形框加入到窗口中
            vis.add_geometry(line_set)

            ##add groud truth
            corners_3d_cam2 = utils.compute_3D_box_cam2(height, width, lenght, x, y, z, rotation)
            calib = utils.Calibration('data/kitti', from_video=True)
            bbox_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])

            # lines_box = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
            # 设置点与点之间线段的颜色
            colors = np.array([[0, 1, 1] for j in range(len(lines_box))])
            # 创建Bbox候选框对象
            line_set = o3d.geometry.LineSet()
            # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            # 设置每条线段的颜色
            line_set.colors = o3d.utility.Vector3dVector(colors)
            # 把八个顶点的空间信息转换成o3d可以使用的数据类型
            line_set.points = o3d.utility.Vector3dVector(bbox_3d_velo)
            # 将矩形框加入到窗口中
            vis.add_geometry(line_set)

    vis.get_render_option().point_size = 1  # 点云大小
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
    vis.run()


if __name__ == "__main__":
    points = "000045"
    visualize(points)
    # show_3dBox()
