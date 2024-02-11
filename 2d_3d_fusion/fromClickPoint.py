import numpy as np
import cv2 as cv
import utils
import open3d as o3d
from skimage.transform import resize

if __name__ == "__main__":
    # 图片路径
    sequence = "000045"
    img = cv.imread('data/kitti/training/image_2/{}.png'.format(sequence))
    a = []
    b = []


    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            a.append(x)
            b.append(y)
            cv.imshow("image", img)



    cv.namedWindow("image")
    cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv.imshow("image", img)
    cv.waitKey(0)
    a = np.asarray(a)
    b = np.asarray(b)
    print(a, b)

    loadData = np.load('kitty_open3d/{}_disp.npy'.format(sequence))
    loadData = np.asarray(resize(loadData.squeeze(), [375, 1242]))
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
    C2_depth, pcloud_C2, pcloud_C0 = utils.velodyne_to_camera_2(pointcloud, calib)
    # ground truth
    for line in labels:
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
            center_2d = [(lx + rx) / 2, (ly + ry) / 2]
            center_depth = 0  # initialize depth of center point
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

    for j in range(0, len(a)):
        lx = int(a[j])
        ly = int(b[j])
        depth = 721 * 0.54 / loadData[ly][lx] / 1000
        point_3d_cam2 = [lx * depth, ly * depth, depth]
        calib= utils.get_sequence_calib(sequence)
        point_3d_cam0 = np.asarray(utils.camera_2_to_camera_0(point_3d_cam2, calib))
        corners_3d_cam2 = utils.compute_3D_box_cam2(height, width, lenght, point_3d_cam0[0], point_3d_cam0[1],
                                                    point_3d_cam0[2], 1.6)
        center_3d=point_3d_cam0
    # for j in range(0, len(a), 2):
    #     distance_lPointlist = []
    #     distance_rPointlist = []
    #     for i in range(len(pcloud_C2)):
    #         distance_lPoint = math.pow(pcloud_C2[i][0] - a[j], 2) + math.pow(pcloud_C2[i][1] - b[j], 2)
    #         distance_lPointlist.append(distance_lPoint)
    #         distance_rPoint = math.pow(pcloud_C2[i][0] - a[j + 1], 2) + math.pow(pcloud_C2[i][1] - b[j + 1], 2)
    #         distance_rPointlist.append(distance_rPoint)
    #     lPoint_depth = C2_depth[np.argmin(distance_lPointlist)]
    #     rPoint_depth = C2_depth[np.argmin(distance_rPointlist)]
    #     index_left = np.argmin(distance_lPointlist)
    #     index_right = np.argmin(distance_rPointlist)
    #     lPoint_3d = [pcloud_C0[index_left][0], pcloud_C0[index_left][1], pcloud_C0[index_left][2]]
    #     rPoint_3d = [pcloud_C0[index_right][0], pcloud_C0[index_right][1], pcloud_C0[index_right][2]]
    #     center_3d = [(lPoint_3d[0] + rPoint_3d[0]) / 2, (lPoint_3d[1] + rPoint_3d[1]) / 2,
    #                  (lPoint_3d[2] + rPoint_3d[2]) / 2]
        # print(center_3d)
        corners_3d_cam2 = utils.compute_3D_box_cam2(1.4, 1.6, 4.1, center_3d[0], center_3d[1], center_3d[2], 1.6)
        # 从cam2坐标转换到velo坐标系 .T转置操作
        calib = utils.Calibration('data/kitti', from_video=True)
        bbox_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
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
        line_set.points = o3d.utility.Vector3dVector(bbox_3d_velo)
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

    vis.get_render_option().point_size = 1  # 点云大小
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
    vis.run()
