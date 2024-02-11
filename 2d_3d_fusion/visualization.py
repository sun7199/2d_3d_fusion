import numpy as np
import open3d as o3d


def draw_bbox(filename, type, color):
    textured_mesh = o3d.io.read_triangle_mesh(
        'F:/detr3d-main/output/{}/{}_{}.obj'.format(filename, filename, type))
    textured_mesh.compute_vertex_normals()
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(textured_mesh.vertices)
    vis.add_geometry(pcobj)

    # obj顶点转array
    textured_pc = np.asarray(textured_mesh.vertices)
    print(textured_pc[0])
    step = 8
    textured_pc = [textured_pc[i:i + step] for i in range(0, len(textured_pc), step)]

    for i in range(0, len(textured_pc)):
        lines_box = np.array([[0, 1], [1, 3], [0, 2], [2, 3], [4, 5], [5, 7], [4, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])

        # lines_box = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])
        # 设置点与点之间线段的颜色
        colors = np.array([color for j in range(len(lines_box))])
        # 创建Bbox候选框对象
        line_set = o3d.geometry.LineSet()
        # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        # 设置每条线段的颜色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # 把八个顶点的空间信息转换成o3d可以使用的数据类型
        line_set.points = o3d.utility.Vector3dVector(textured_pc[i])
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

vis = o3d.visualization.Visualizer()
filename='n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151606048630'
# obj面片显示
textured_mesh = o3d.io.read_triangle_mesh(
    'F:/detr3d-main/output/{}/{}_points.obj'.format(filename,filename))
textured_mesh.compute_vertex_normals()

# obj顶点显示
pcobj = o3d.geometry.PointCloud()
pcobj.points = o3d.utility.Vector3dVector(textured_mesh.vertices)
vis.create_window()
vis.add_geometry(pcobj)
draw_bbox(filename,'pred',[0,1,0])
draw_bbox(filename,'gt',[1,0,0])
vis.get_render_option().point_size = 1  # 点云大小
vis.get_render_option().background_color = np.asarray([0, 0, 0])  # 背景颜色
vis.run()



