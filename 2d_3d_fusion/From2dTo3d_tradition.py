import numpy as np
import cv2 as cv

from skimage.transform import resize
from scipy.linalg import solve

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


# 函数参数为左右相片同名点的像素坐标，获取方式后面介绍
# lx，ly为左相机某点像素坐标，rx，ry为右相机对应点像素坐标
def uvToXYZ(lx, ly, rx, ry):
    mLeft = np.hstack([leftRotation, leftTranslation])
    rect_leftIntrinsic = np.dot(leftIntrinsic, R2_rect)
    mLeftM = np.dot(rect_leftIntrinsic, mLeft)
    mRight = np.hstack([rightRotation, rightTranslation])
    rect_rightIntrinsic = np.dot(rightIntrinsic, R3_rect)
    mRightM = np.dot(rect_rightIntrinsic, mRight)
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
    XYZ = calib.project_ref_to_velo(XYZ)
    translation=np.array([4*4.069766e-01,2*7.631618e-01,-4*2.717806e-01])
    # rotation=np.array([[7.533745e-03,-9.999714e-01,-6.166020e-04 ],
    #                   [1.480249e-02,7.280733e-04,-9.998902e-01],
    #                   [9.998621e-01,7.523790e-03,1.480755e-02]])
    # XYZ=np.transpose(np.dot(rotation,XYZ.T))
    XYZ=XYZ+translation
    # 从cam2坐标转换到velo坐标系 .T转置操作
    return XYZ


def visualize(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='32line Lidar', width=1080, height=720, left=300, top=150, visible=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3].reshape(-1, 3))
    calib = utils.get_sequence_calib()
    corners_3d_cam2 = utils.transform_3dbox_to_image([1.47, 1.63, 4.11], [-8.95, 1.92, 15.21], 1.59, calib, 'P2')
    corners_3d_cam3 = utils.transform_3dbox_to_image([1.47, 1.63, 4.11], [-8.95, 1.92, 15.21], 1.59, calib, 'P3')
    # 从cam2坐标转换到velo坐标系 .T转置操作
    x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
    corners_3d_velo = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])

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
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    corners_3d_cam2 = utils.transform_3dbox_to_image([1.80 , 1.61, 3.83], [8.66, 1.94, 21.96], 1.60, calib, 'P2')
    corners_3d_cam3 = utils.transform_3dbox_to_image([1.80 , 1.61, 3.83], [8.66, 1.94, 21.96], 1.60, calib, 'P3')
    # 从cam2坐标转换到velo坐标系 .T转置操作
    x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
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

    corners_3d_cam2 = utils.transform_3dbox_to_image([1.45 , 1.45 , 3.40], [8.46 , 1.99 , 27.06], 1.65, calib, 'P2')
    corners_3d_cam3 = utils.transform_3dbox_to_image([1.45 , 1.45 , 3.40], [8.46 , 1.99 , 27.06], 1.65, calib, 'P3')
    x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
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

    # corners_3d_cam2 = utils.transform_3dbox_to_image([1.75 ,1.71 ,4.31], [8.27, 2.02 ,31.88], 1.61, calib, 'P2')
    # corners_3d_cam3 = utils.transform_3dbox_to_image([1.75 ,1.71 ,4.31], [8.27, 2.02 ,31.88], 1.61, calib, 'P3')
    # x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    # x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    # x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    # x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    # x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    # x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    # x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    # x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
    # corners_3d_velo = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
    #
    # # corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    #
    # lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
    #                       [0, 4], [1, 5], [2, 6], [3, 7]])
    #
    # # lines_box = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
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
    #
    corners_3d_cam2 = utils.transform_3dbox_to_image([1.53, 1.66, 4.01], [7.54, 2.03 , 41.53], 1.62, calib, 'P2')
    corners_3d_cam3 = utils.transform_3dbox_to_image([1.53, 1.66, 4.01], [7.54, 2.03 , 41.53], 1.62, calib, 'P3')
    x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
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
    #
    # corners_3d_cam2 = utils.transform_3dbox_to_image([1.48, 1.60 ,3.90], [7.16, 2.05 ,46.70], 1.64, calib, 'P2')
    # corners_3d_cam3 = utils.transform_3dbox_to_image([1.48, 1.60 ,3.90], [7.16, 2.05 ,46.70], 1.64, calib, 'P3')
    # x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    # x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    # x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    # x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    # x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    # x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    # x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    # x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
    # corners_3d_velo = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])

    # corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)

    corners_3d_cam2 = utils.transform_3dbox_to_image([3.33, 2.37, 23.29], [11.16, 1.55 , 79.53], -1.51, calib, 'P2')
    corners_3d_cam3 = utils.transform_3dbox_to_image([3.33, 2.37, 23.29], [11.16, 1.55 , 79.53], -1.51, calib, 'P3')
    x1 = uvToXYZ(corners_3d_cam2[0][0], corners_3d_cam2[0][1], corners_3d_cam3[0][0], corners_3d_cam3[0][1])
    x2 = uvToXYZ(corners_3d_cam2[1][0], corners_3d_cam2[1][1], corners_3d_cam3[1][0], corners_3d_cam3[1][1])
    x3 = uvToXYZ(corners_3d_cam2[2][0], corners_3d_cam2[2][1], corners_3d_cam3[2][0], corners_3d_cam3[2][1])
    x4 = uvToXYZ(corners_3d_cam2[3][0], corners_3d_cam2[3][1], corners_3d_cam3[3][0], corners_3d_cam3[3][1])
    x5 = uvToXYZ(corners_3d_cam2[4][0], corners_3d_cam2[4][1], corners_3d_cam3[4][0], corners_3d_cam3[4][1])
    x6 = uvToXYZ(corners_3d_cam2[5][0], corners_3d_cam2[5][1], corners_3d_cam3[5][0], corners_3d_cam3[5][1])
    x7 = uvToXYZ(corners_3d_cam2[6][0], corners_3d_cam2[6][1], corners_3d_cam3[6][0], corners_3d_cam3[6][1])
    x8 = uvToXYZ(corners_3d_cam2[7][0], corners_3d_cam2[7][1], corners_3d_cam3[7][0], corners_3d_cam3[7][1])
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


def show_3dBox():
    calib = utils.get_sequence_calib()
    x1 = uvToXYZ(164, 278, 134, 278)
    x2 = uvToXYZ(75, 277, 46, 278)
    x3 = uvToXYZ(206, 252, 183, 253)
    x4 = uvToXYZ(273, 253, 251, 253)
    x5 = uvToXYZ(164, 197, 134, 197)
    x6 = uvToXYZ(75, 197, 46, 197)
    x7 = uvToXYZ(206, 191, 183, 191)
    x8 = uvToXYZ(273, 191, 251, 191)
    corners_3d_img = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
    corners_3d_img = corners_3d_img.astype(int)
    img = cv.imread('data/kitti/training/image_3/000045.png')
    bbox_color = (48, 96, 48)
    thickness = 2

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

    # draw front lines
    cv.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]),
            (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
    cv.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]),
            (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

    cv.imshow('3dbox', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # points = np.fromfile("data/kitti/training/velodyne/000055.bin", np.float32).reshape(-1, 4)
    # visualize(points)
    # show_3dBox()
    loadData = np.load('kitty_open3d/000045_disp.npy')
    loadData= resize(loadData.squeeze(), [375,1242])
    print(loadData[205][517]*721)
    cv.imshow('111',loadData)
    cv.waitKey(0)
    img=cv.imread('kitty_open3d/000045_disp.png')
    img1=cv.imread('F:/data/kitti/training/image_2/000045.png')
    print(img.shape)
    print(img1)
    dx=7.215377000000e+02
    dy=7.215377000000e+02
    u0=6.095593000000e+02
    v0=1.728540000000e+02
    i=273
    j=277
    A=np.array([[loadData[277][273],1],[loadData[242][375],1]])
    B=np.array([15.21,21.96])
    result=solve(A,B)
    print(721*0.54/loadData[277][273]/1000)
    print(721*0.54/loadData[242][375])
    print(721*0.54/loadData[229][420])
    print(721*0.54/loadData[221][455])
    print(721*0.54/loadData[210][501])
    print(721*0.54/loadData[205][518])
    result=-766.39901498*loadData[205][518]+34
    calib= utils.get_sequence_calib("000045")
    print(calib["P2"])
    p2_inverse=inverse_rigid_trans(calib["P2"])
    image_p2=np.asarray([[273*15.21,277*15.21,15.21]])
    rotation = np.asarray(calib['P2'][0:3, 0:3])
    translation = np.asarray(calib['P2'][0:3, 3])
    centerPoint = np.dot((image_p2 - translation), np.linalg.inv(rotation.T))
    print(centerPoint)