import numpy as np


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

def transform_3dbox_to_pointcloud(dimension, location, rotation):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # from camera coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

    return corners_3d


def transform_corordinate_to_3dbox(dimension, location):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # shift the corners to from origin to location
    corners_3d = corners_3d.T + np.array([x, y, z])

    return corners_3d


def transform_3dbox_to_image(dimension, location, rotation, calib, cam):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, lenght = dimension
    x, y, z = location
    x_corners = [lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # only show 3D bounding box for objects in front of the camera
    if np.any(corners_3d[:, 2] < 0.1):
        corners_3d_img = None
    else:
        # from camera coordinate to image coordinate
        corners_3d_temp = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_3d_img = np.matmul(corners_3d_temp, calib[cam].T)

        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]

    return corners_3d_img


def get_sequence_calib(sequence):
    """
    get the calib parameters
    :param sequence_name: sequence name
    :return: calib
    """

    # load data
    sequence_calib_path = 'F:/data/kitti/training/calib/{}.txt'.format(sequence)
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


def velodyne_to_camera_2(pcloud, calib):
    pcloud_temp = np.hstack((pcloud[:, :3], np.ones((pcloud.shape[0], 1), dtype=np.float32)))  # [N, 4]
    pcloud_C0 = np.dot(pcloud_temp, np.dot(calib['Tr_velo_cam'].T, calib['Rect'].T))  # [N, 3]

    pcloud_C0_temp = np.hstack((pcloud_C0, np.ones((pcloud.shape[0], 1), dtype=np.float32)))
    pcloud_C2 = np.dot(pcloud_C0_temp, calib['P2'].T)  # [N, 3]

    pcloud_C2_depth = pcloud_C2[:, 2]
    for i in range(len(pcloud_C2_depth)):
        if pcloud_C2_depth[i]<0:
            pcloud_C2_depth[i]=-1000
    pcloud_C2 = (pcloud_C2[:, :2].T / pcloud_C2[:, 2]).T

    return pcloud_C2_depth, pcloud_C2,pcloud_C0

def velodyne_to_camera_3(pcloud, calib):
    pcloud_temp = np.hstack((pcloud[:, :3], np.ones((pcloud.shape[0], 1), dtype=np.float32)))  # [N, 4]
    pcloud_C0 = np.dot(pcloud_temp, np.dot(calib['Tr_velo_cam'].T, calib['Rect'].T))  # [N, 3]

    pcloud_C0_temp = np.hstack((pcloud_C0, np.ones((pcloud.shape[0], 1), dtype=np.float32)))
    pcloud_C3 = np.dot(pcloud_C0_temp, calib['P3'].T)  # [N, 3]

    pcloud_C3_depth = pcloud_C3[:, 2]
    for i in range(len(pcloud_C3_depth)):
        if pcloud_C3_depth[i]<0:
            pcloud_C3_depth[i]=-1000
    pcloud_C3 = (pcloud_C3[:, :2].T / pcloud_C3[:, 2]).T

    return pcloud_C3_depth, pcloud_C3,pcloud_C0

def camera_2_to_camera_0(cam2, calib):
    image_p2 = cam2
    rotation = np.asarray(calib['P2'][0:3, 0:3])
    translation = np.asarray(calib['P2'][0:3, 3])
    centerPoint = np.dot((image_p2 - translation), np.linalg.inv(rotation.T))
    return centerPoint

# cam2坐标下的 3D box

def compute_3D_box_cam2(h, w, l, x, y, z, yaw):
    '''
    Return:3Xn in cam2 coordinate
    '''
    # 建立旋转矩阵R
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    # 计算8个顶点坐标
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # 使用旋转矩阵变换坐标
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # 最后在加上中心点
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2




class Calibration(object):
    ''' Calibration matrices and utils8
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file('data/kitti/calib_cam_to_cam.txt')
        velo2cam = self.read_calib_file('data/kitti/calib_velo_to_cam.txt')
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr


# for debug
if __name__ == "__main__":
    sequence="000045"

    points = np.fromfile("data/kitti/training/velodyne/000045.bin", np.float32).reshape(-1, 4)

    calib = get_sequence_calib(sequence)
    pcloud_C2_depth, pcloud_C2,pcloud_C0=velodyne_to_camera_2(points,calib)
    print(pcloud_C2)
    corners_3d_cam2 = transform_3dbox_to_image([1.47 ,1.63 ,4.11], [-8.95 ,1.92, 15.21], 1.62,calib,'P2')
    print(corners_3d_cam2)
    pcloud_C2=pcloud_C2.astype(int)
    for i in range(len(pcloud_C2)):
        if pcloud_C2[i][0]==75 and pcloud_C2[i][1]==192:
            print(i)

    # 从cam2坐标转换到velo坐标系
    calib = Calibration('data/kitti', from_video=True)
    corners_3d_cam2 = compute_3D_box_cam2(1.80, 1.61, 3.83, -8.66, 1.94, 21.96, 1.60)
    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T)
    print(corners_3d_velo)
    # pcloud_C2_depth, pcloud_C2,pcloud_C0=velodyne_to_camera_2(points,calib)
    # pcloud_C3_depth, pcloud_C3, pcloud_C0 = velodyne_to_camera_2(points, calib)
    # print(pcloud_C0)
    # np.savetxt("3.txt",points)
    # a=[]
    # for i in range(len(pcloud_C2)):
    #     if (pcloud_C2.astype(int)[i][0]==75)and(pcloud_C2.astype(int)[i][1]==192):
    #         a.append(i)
    # print(a)

