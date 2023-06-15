import open3d as o3d
import numpy as np
import pyVHACD

def process_point_cloud_to_vhacd(pcd):
    # 使用BPA算法进行三维重构
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)

    # # 将mesh模型的点和面转换为numpy array
    # vertices = np.asarray(mesh.vertices)
    # triangles = np.asarray(mesh.triangles)

    # # 使用pyVHACD计算convex hull
    # vhacd_model = pyVHACD.compute_vhacd(vertices, triangles)

    # # 将convex hull转换为open3d的mesh模型
    # vertices = []
    # faces = []
    # for i, (per_vertices, per_triangles) in enumerate(vhacd_model):
    #     vertices.append(per_vertices)
    #     faces.append(per_triangles)
    # vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    # faces = np.concatenate(faces, axis=0).astype(np.int32)
    # faces = faces.reshape(len(faces), 3)

    # # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)

    return mesh

# 用一个示例点云文件来测试该函数
points = np.load("/home/huangdehao/github_projects/pybullet_franka/0_pc.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
vhacd_mesh = process_point_cloud_to_vhacd(pcd)
o3d.visualization.draw_geometries([vhacd_mesh])
