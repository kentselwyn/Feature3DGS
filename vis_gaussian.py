import torch
import numpy as np
import open3d as o3d
from scene import GaussianModel
from sklearn.cluster import DBSCAN



def choose_th(score, histogram_th):
    score_flat = score.flatten()
    percentile_value = torch.quantile(score_flat, float(histogram_th))

    return percentile_value.item()


# python vis_gaussian.py
if __name__=="__main__":
    gaussians = GaussianModel(3)
    gaussians.load_ply("/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/pgt_7scenes_stairs/outputs/SP_imrate:1_th:0.01_mlpdim:16_kptnum:1024_score0.6_rgb/point_cloud/iteration_30000/point_cloud.ply")


    positions = gaussians.get_xyz
    scores = gaussians.get_score_feature.squeeze(-1)

    percentage = 0.92
    th = choose_th(scores, percentage)

    print(positions.shape)
    print(scores.shape)
    print(th)
    mask_score = scores>th
    sum_score = mask_score.sum()
    mask_score = mask_score.squeeze(-1)
    print(sum_score)
    # breakpoint()

    scores = scores[mask_score]
    filtered_points = positions[mask_score]


    # breakpoint()
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    filtered_points = filtered_points.cpu().detach().numpy()
    scores_np = scores.cpu().detach().numpy()

    colors = np.zeros((scores_np.shape[0], 3))  # Initialize RGB colors
    colors[:, 0] = scores_np[:, 0]


    dbscan = DBSCAN(eps=0.004, min_samples=8)
    labels = dbscan.fit_predict(filtered_points)


    # Find cluster centers
    unique_labels = set(labels)
    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise
        cluster_points = filtered_points[labels == label]
        center = cluster_points.mean(axis=0)  # Compute center
        cluster_centers.append(center)

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # o3d.io.write_point_cloud("pointcloud_with_scores.ply", point_cloud)
    # print("Point cloud saved to 'pointcloud_with_scores.ply'")
        
    # Print the cluster centers
    # print("Cluster centers:", cluster_centers, len(cluster_centers))

    # Visualize clusters and centers using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    colors = np.array([[1, 0, 0] if lbl != -1 else [0.5, 0.5, 0.5] for lbl in labels])  # Red for clusters, grey for noise
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f"clusters_{percentage}.ply", pcd)


    center_points = np.array(cluster_centers)
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector(center_points)
    center_pcd.paint_uniform_color([1, 0, 0])  # Green for cluster centers
    o3d.io.write_point_cloud(f"cluster_centers_{percentage}.ply", center_pcd)

    # o3d.visualization.draw_geometries([pcd, center_pcd])


