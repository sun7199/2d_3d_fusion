import numpy as np
import open3d as o3d
import utils
import cv2 as cv


def visualize(velodyneFile):
    pointcloud = np.fromfile(velodyneFile, np.float32).reshape(-1, 4)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='ground truth', width=1080, height=720, left=300, top=150, visible=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3].reshape(-1, 3))
    vis.add_geometry(point_cloud)
    calib = utils.Calibration('data/kitti', from_video=True)
    #  转换成相机坐标系
    corners_3d_cam2 = utils.compute_3D_box_cam2(1.48, 1.35, 3.93, -23.47, 2.44, 32.12, 1.62)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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

    corners_3d_cam2 = utils.compute_3D_box_cam2(1.80, 1.61, 3.83, -8.66, 1.94, 21.96, 1.60)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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

    corners_3d_cam2 = utils.compute_3D_box_cam2(1.45, 1.45, 3.40, -8.46, 1.99, 27.06, 1.65)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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

    corners_3d_cam2 = utils.compute_3D_box_cam2(1.75, 1.71, 4.31, -8.27, 2.02, 31.88, 1.61)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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

    corners_3d_cam2 = utils.compute_3D_box_cam2(1.53, 1.66, 4.01, -7.54, 2.03, 41.53, 1.62)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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
   # eight points for kitti label cordinate
    corners_3d_cam2 = utils.compute_3D_box_cam2(1.48, 1.35, 3.93, -23.47, 2.44, 32.12, 1.62)
    print(corners_3d_cam2.T)
    # 从cam2坐标转换到velo坐标系 .T转置操作
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
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

    # corners_3d_cam2 = utils.compute_3D_box_cam2(3.33 ,2.37, 23.29 ,11.16, 1.55, 79.53 ,-1.51)
    # # 从cam2坐标转换到velo坐标系 .T转置操作
    # corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    # lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
    #                       [0, 4], [1, 5], [2, 6], [3, 7]])
    # # 设置点与点之间线段的颜色
    # colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    # # 创建Bbox候选框对象
    # line_set = o3d.geometry.LineSet()
    # # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
    # line_set.lines = o3d.utility.Vector2iVector(lines_box)
    # # 设置每条线段的颜色
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # # 把八个顶点的空间信息转换成o3d可以使用的数据类型
    # line_set.points = o3d.utility.Vector3dVector(corners_3d_velo)
    # # 将矩形框加入到窗口中
    # vis.add_geometry(line_set)

    vis.get_render_option().point_size = 1  # 点云大小
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
    vis.run()

def show_3dBox_image(sequence):
    sequence_label_path = 'data/kitti/training/label_2/{}.txt'.format(sequence)
    with open(sequence_label_path, 'r') as f:
        labels = f.readlines()
    calib = utils.get_sequence_calib(sequence)
    img = cv.imread('data/kitti/training/image_2/{}.png'.format(sequence))
    bbox_color = (48, 96, 48)
    thickness = 2
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
            corners_3d_img = utils.transform_3dbox_to_image([height, width, lenght], [x, y, z], rotation, calib, 'P2')
            corners_3d_img = corners_3d_img.astype(int)
            print(corners_3d_img)

        # draw lines in the image
        # p10-p1, p1-p2, p2-p3, p3-p0
            cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                    (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                    (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                    (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                    (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

            # p4-p5, p5-p6, p6-p7, p7-p0
            cv.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                    (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                    (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                    (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                    (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

            # p0-p4, p1-p5, p2-p6, p3-p7
            cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                    (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                    (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                    (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
            cv.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                    (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

            cv.line(img, (75 ,192),
                    (273 ,277), color=(96,48,96), thickness=thickness)

    cv.imshow('3dbox', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # visualize("data/kitti/training/velodyne/000055.bin")
    sequence="000045"
    show_3dBox_image(sequence)
