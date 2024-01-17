import cv2
import numpy as np
import open3d as o3d

data_path = "/home/huangdehao/github_projects/anygrasp_sdk/grasp_detection/example_data_me/textured_simple.obj"
mesh = o3d.io.read_triangle_mesh(data_path, True)
# o3d.visualization.draw_geometries([mesh])

# # point_cloud = mesh.sample_points_uniformly(number_of_points=10000)
# # point_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = mesh.vertices
# point_cloud.colors = mesh.vertex_colors
# point_cloud.normals = mesh.vertex_normals
# # points = np.asarray(point_cloud.points).astype(np.float32)
# # colors = np.asarray(point_cloud.colors).astype(np.float32)
# # print(points.min(axis=0), points.max(axis=0))
# o3d.visualization.draw_geometries([point_cloud])

point_cloud = mesh.sample_points_poisson_disk(number_of_points=3000)

# Compute the colors for the point cloud
# point_cloud.colors = o3d.utility.Vector3dVector(
#     [mesh.textures[0].evaluate(uv) for uv in mesh.triangle_uvs]
# )

# Save or visualize the point cloud
# o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
o3d.visualization.draw_geometries([point_cloud])
pass