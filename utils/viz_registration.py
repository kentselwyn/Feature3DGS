import os
import numpy as np
import open3d as o3d
import os.path as osp

from utils.open3d_utils import *

K = 30 # Number of visualized scenes
RESULT_ROOT = osp.join(".", "fine-experiment-cosine") # Root to registration output directory

def visualize(scene_id):
    pointcloud_1 = o3d.io.read_point_cloud(osp.join(scene_id, "before_ref.ply"))
    pointcloud_2 = o3d.io.read_point_cloud(osp.join(scene_id, "before_src.ply"))

    correspondences = np.load(osp.join(scene_id, "correspondence.npz"))
    sorted_index = np.argsort(correspondences["similarity"], axis=-1)[::-1][:K]
    correspondence_lines = make_open3d_corr_lines(
        ref_corr_points=correspondences["ref"][sorted_index],
        src_corr_points=correspondences["src"][sorted_index],
        label="pos"
    )

    # Visualizes correspondence
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0]
    vis.add_geometry(pointcloud_1)
    vis.add_geometry(pointcloud_2)
    vis.add_geometry(correspondence_lines)
    vis.create_window(window_name='Cosine Correspondence', width=int(1920/2), height=1080, left=int(1920/2), top=0, visible=True)
    vis.run()

    # Visualizes registration result
    pointcloud_1 = o3d.io.read_point_cloud(osp.join(scene_id, "after_ref.ply"))
    pointcloud_2 = o3d.io.read_point_cloud(osp.join(scene_id, "after_src.ply"))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0, 0, 0]
    vis.add_geometry(pointcloud_1)
    vis.add_geometry(pointcloud_2)
    vis.create_window(window_name='Result', width=int(1920/2), height=1080, left=int(1920/2), top=0, visible=True)
    vis.run()

if __name__ == "__main__":
    scene_ids = os.listdir(RESULT_ROOT)
    for scene_id in scene_ids[:K]:
        if osp.isdir(osp.join(RESULT_ROOT, scene_id)):
            visualize(osp.join(RESULT_ROOT, scene_id))