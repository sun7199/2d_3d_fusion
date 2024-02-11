import math

import numpy as np
import cv2 as cv
import utils
import open3d as o3d


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''

    inv_Tr = np.zeros_like(Tr)  # 3x4

    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])

    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])

    return inv_Tr


# 左/右相机内参数、旋转、平移矩阵
# 左/右相机内参数、旋转、平移矩阵
p0 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
               [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
               [0.000000e+00, 0.000000e+00, 1.000000e+00]])

p2 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
               [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

p3 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02],
               [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.199936000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.729905000000e-03]])

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

# leftTranslation = np.array([[2.573699e-16] ,
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

R0_rect = np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0],
                    [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0],
                    [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0],
                    [0, 0, 0, 1]])

R3_rect = np.array([[9.998321e-01, -7.193136e-03, 1.685599e-02],
                    [7.232804e-03, 9.999712e-01, -2.293585e-03],
                    [-1.683901e-02, 2.415116e-03, 9.998553e-01]])


# 函数参数为左右相片同名点的像素坐标，获取方式后面介绍
# lx，ly为左相机某点像素坐标，rx，ry为右相机对应点像素坐标
def uvToXYZ(lx, ly, rx, ry):
    mLeft = np.hstack([leftRotation, leftTranslation])
    rect_leftIntrinsic = np.dot(leftIntrinsic, R2_rect)
    mLeftM = np.dot(p2, R0_rect)
    mRight = np.hstack([rightRotation, rightTranslation])
    rect_rightIntrinsic = np.dot(rightIntrinsic, R3_rect)
    mRightM = np.dot(p3, R0_rect)
    A = np.zeros(shape=(4, 3))
    for i in range(0, 3):
        A[0][i] = lx * mLeftM[2, i] - mLeftM[0][i]
    for i in range(0, 3):
        A[1][i] = ly * mLeftM[2][i] - mLeftM[1][i]
    for i in range(0, 3):
        A[2][i] = rx * mRightM[2][i] - mRightM[0][i]
    for i in range(0, 3):
        A[3][i] = ry * mRightM[2][i] - mRightM[1][i]
    B = np.zeros(shape=(4, 1))
    for i in range(0, 2):
        B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
    for i in range(2, 4):
        B[i][0] = mRightM[i - 2][3] - rx * mRightM[2][3]
    XYZ = np.zeros(shape=(3, 1))
    # 根据大佬的方法，采用最小二乘法求其空间坐标
    cv.solve(A, B, XYZ, cv.DECOMP_SVD)
    XYZ = XYZ.reshape(1, 3)
    calib = utils.Calibration('data/kitti', from_video=True)

    # transpose=np.array([[1,0,0],
    #                      [0,0,1],
    #                      [0,1,0]])
    # XYZ=np.dot(XYZ,transpose)
    XYZ = calib.project_rect_to_velo(XYZ)
    translation = np.array([4.069766e-01, 20 * 7.631618e-01, -4 * 2.717806e-01])
    # rotation=np.array([[7.533745e-03,-9.999714e-01,-6.166020e-04 ],
    #                   [1.480249e-02,7.280733e-04,-9.998902e-01],
    #                   [9.998621e-01,7.523790e-03,1.480755e-02]])
    # XYZ=np.transpose(np.dot(rotation,XYZ.T))
    XYZ = XYZ
    # 从cam2坐标转换到velo坐标系 .T转置操作
    return XYZ


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
    calib= utils.get_sequence_calib(sequence)
    C2_depth, pcloud_C2,pcloud_C0 = utils.velodyne_to_camera_2(pointcloud, calib)
    C3_depth, pcloud_C3, pcloud_C0 = utils.velodyne_to_camera_3(pointcloud, calib)
    pcloud_C2=pcloud_C2.astype(int)
    pcloud_C3 = pcloud_C3.astype(int)
    # loadData = np.load('kitty_open3d/{}_disp.npy'.format(sequence))
    # loadData = np.asarray(resize(loadData.squeeze(), [375, 1242]))

    # get bbox according to txt label
    for line in labels:
        Point_velodyne = []
        line = line.split()
        object_type, truncat, occ, alpha, lx, ly, rx, ry, height, width, lenght, x, y, z, rotation = line

        lx=float(lx)
        ly=float(ly)
        rx=float(rx)
        ry=float(ry)
        height = float(height)
        width = float(width)
        lenght = float(lenght)
        x = float(x)
        y = float(y)
        z = float(z)
        rotation = float(rotation)
        if object_type != 'DontCare':
            calib = utils.get_sequence_calib(sequence)
            corners_3d_img = utils.transform_3dbox_to_image([height, width, lenght], [x, y, z], rotation, calib, 'P2')
            corners_3d_img=corners_3d_img.astype(int)
            points_cam2_3d=[]
            # for i in range(len(corners_3d_img)):
            #     lx=int(corners_3d_img[i][0])
            #     ly=int(corners_3d_img[i][1])
                # depth=721 * 0.54 / loadData[ly][lx]/1000
                # point_3d=[lx*depth,ly*depth,depth]
                # points_cam2_3d.append(point_3d)
            center_2d=[(corners_3d_img[2][0]+corners_3d_img[7][0])/2,(corners_3d_img[2][1]+corners_3d_img[7][1])/2]
            # lx=int(center_2d[0])
            # ly=int(center_2d[1])
            # depth = 721 * 0.54 / loadData[ly][lx] / 1242
            # point_3d_cam2 = [lx * depth, ly * depth, depth]
            distance_list=[]
            for j in range(len(pcloud_C2)):
                distance=(math.pow(pcloud_C2[j][0]-center_2d[0],2)+math.pow(pcloud_C2[j][1]-center_2d[1],2)+
                          math.pow(pcloud_C3[j][0] - center_2d[0], 2) + math.pow(pcloud_C3[j][1] - center_2d[1], 2))
                distance_list.append(distance)
            index=np.argmin(distance_list)
            point_C0=[pcloud_C0[index][0],pcloud_C0[index][1],pcloud_C0[index][2]]
            point_3d_cam0=np.array(point_C0)
            # Point_velodyne.append(point_C0)
            points_cam0_3d=[]
            # for i in range(len(points_cam2_3d)):
            #     point_3d=utils.camera_2_to_camera_0(points_cam2_3d[i],calib)
            #     points_cam0_3d.append(point_3d)
            # point_3d_cam0 = np.asarray(utils.camera_2_to_camera_0(point_3d_cam2, calib))
            corners_3d_cam2 = utils.compute_3D_box_cam2(height, width, lenght, point_3d_cam0[0], point_3d_cam0[1], point_3d_cam0[2], rotation)
            # 从cam2坐标转换到velo坐标系 .T转置操作
            # points_cam0_3d=np.asarray(points_cam0_3d)
            calib = utils.Calibration('data/kitti', from_video=True)
            box_3d_velodyne = calib.project_rect_to_velo(corners_3d_cam2.T)
        ###########最小二乘法
        # corners_3d_cam2 = utils.transform_3dbox_to_image([height, width, lenght], [x, y, z], rotation, calib, 'P2')
        # corners_3d_cam3 = utils.transform_3dbox_to_image([height, width, lenght], [x, y, z], rotation, calib, 'P3')
        # # 从cam2坐标转换到velo坐标系 .T转置操作
        # if corners_3d_cam2 is not None:
        #     center = uvToXYZ((corners_3d_cam2[1][0] + corners_3d_cam2[3][0]) / 2,
        #                      (corners_3d_cam3[7][1] + corners_3d_cam3[1][1]) / 2,
        #                      (corners_3d_cam2[1][0] + corners_3d_cam2[3][0]) / 2,
        #                      (corners_3d_cam3[7][1] + corners_3d_cam3[1][1]) / 2)
        #     bbox = utils.transform_corordinate_to_3dbox([1.47, 1.63, 4.11], center[0])

            # print(Point_velodyne)
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
            line_set.points = o3d.utility.Vector3dVector(box_3d_velodyne)
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
    points = "000085"
    visualize(points)
    # show_3dBox()
