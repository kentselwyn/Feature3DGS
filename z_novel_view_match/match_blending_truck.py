import os
import random
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene import Scene
from os import makedirs
from copy import deepcopy
import torch.nn.functional as F
from argparse import ArgumentParser
from render import  feature_visualize_saving
from Feature3DGS.gaussian_renderer.__init__edit import GaussianModel, render
from utils.scoremap_vis import one_channel_vis
from arguments import ModelParams, PipelineParams, get_combined_args

random.seed(20)

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T


def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w



def get_pose_estimation(obs_view, delta):
    gt_pose_c2w=combine_3dgs_rotation_translation(obs_view.R,obs_view.T)
    start_pose_c2w =  trans_t_xyz(delta[3],delta[4],delta[5]) @ \
                        rot_phi(delta[0]/180.*np.pi) @ rot_theta(delta[1]/180.*np.pi) @ rot_psi(delta[2]/180.*np.pi)  @ \
                            gt_pose_c2w
    start_pose_w2c=torch.from_numpy(np.linalg.inv(start_pose_c2w)).float()
    view = deepcopy(obs_view)
    view.world_view_transform = start_pose_w2c.transpose(0, 1).cuda()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    # view.extrinsic_matrix = start_pose_w2c
    return view, start_pose_w2c


def render_novel_views(idx, view, gaussians, pipe_param, background, novel_paths, delta, num):
    obs_view, new_pose_w2c = get_pose_estimation(view, delta)
    render_pkg = render(obs_view, gaussians, pipe_param, background)

    ############ image ############
    torchvision.utils.save_image(render_pkg["render"], os.path.join(novel_paths["img_renders"], '{0:05d}'.format(idx) + f"_{num}.png"))
    


    ############ feature map ############
    feature_map = F.interpolate(render_pkg["feature_map"].unsqueeze(0), 
                                    size=(view.semantic_feature.shape[1], view.semantic_feature.shape[2]), 
                                    mode='bilinear', align_corners=True).squeeze(0)
    # save visual feature
    feature_map_vis = feature_visualize_saving(feature_map)
    Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(novel_paths["feature_renders"], '{0:05d}'.format(idx) + f"_{num}_feature_vis.png"))
    # save feature map
    feature_map = feature_map.cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(feature_map).half(), 
                os.path.join(novel_paths["feature_tensors"], '{0:05d}'.format(idx) + f"_{num}_fmap.pt"))
    


    ############ score map ############
    score_map = F.interpolate(render_pkg['score_map'].unsqueeze(0), 
                              size=(view.score_feature.shape[1], view.score_feature.shape[2]), 
                              mode='bilinear', align_corners=True).squeeze(0)
    
    score_map_vis = one_channel_vis(score_map)
    score_map_vis.save(os.path.join(novel_paths["score_renders"], '{0:05d}'.format(idx) + f"_{num}_score_vis.png"))
    
    # save feature map
    score_map = score_map.cpu().numpy().astype(np.float16)
    torch.save(torch.tensor(score_map).half(), 
               os.path.join(novel_paths["score_tensors"], '{0:05d}'.format(idx) + f"_{num}_smap.pt"))
    
    ############ intrinsic, extrinsic ############
    T0 = obs_view.extrinsic_matrix
    T1 = new_pose_w2c.numpy()
    T_0to1 = np.matmul(T1, np.linalg.inv(T0))
    K = obs_view.intrinsic_matrix

    np.save(os.path.join(novel_paths["extrinsic"], '{0:05d}'.format(idx) + f"_{num}.npy"), T_0to1)
    np.save(os.path.join(novel_paths["intrinsic"], '{0:05d}'.format(idx) + f"_{num}.npy"), K)
    



def render_test_pairs(model_path, name, iteration, views, 
                      gaussians, pipe_param, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "image_gt")
    
    feature_renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_renders")
    gt_feature_renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_gt")
    feature_tensors_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_tensors")

    score_renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_renders")
    gt_score_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_gt")
    score_tensors_path = os.path.join(model_path, name, "ours_{}".format(iteration), "score_tensors")

    intrinsic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "intrinsics")
    extrinsic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "extrinsics")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    makedirs(feature_renders_path, exist_ok=True)
    makedirs(gt_feature_renders_path, exist_ok=True)
    makedirs(feature_tensors_path, exist_ok=True)

    makedirs(score_renders_path, exist_ok=True)
    makedirs(gt_score_map_path, exist_ok=True)
    makedirs(score_tensors_path, exist_ok=True)

    makedirs(intrinsic_path, exist_ok=True)
    makedirs(extrinsic_path, exist_ok=True)

    novel_paths = {
        'img_renders': render_path,
        'feature_renders': feature_renders_path,
        'feature_tensors': feature_tensors_path,
        'score_renders': score_renders_path,
        'score_tensors': score_tensors_path,
        'intrinsic': intrinsic_path,
        'extrinsic': extrinsic_path
    }

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipe_param, background)

        gt = view.original_image[0:3, :, :]
        gt_feature_map = view.semantic_feature.cuda() 
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        ############## feature map
        feature_map = render_pkg["feature_map"][:16]
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), 
                                    mode='bilinear', align_corners=True).squeeze(0) ###

        # save visual feature
        feature_map_vis = feature_visualize_saving(feature_map)
        Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(feature_renders_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
        
        # save visual gt feature
        gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
        Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(
            os.path.join(gt_feature_renders_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

        # save feature map
        feature_map = feature_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(feature_map).half(), os.path.join(feature_tensors_path, '{0:05d}'.format(idx) + "_fmap.pt"))
        #############

        ############# score map
        score_map = render_pkg['score_map']

        gt_score_map = view.score_feature.cuda()
        score_map = F.interpolate(score_map.unsqueeze(0), size=(gt_score_map.shape[1], gt_score_map.shape[2]), 
                                  mode='bilinear', align_corners=True).squeeze(0) ###
        
        score_map_vis = one_channel_vis(score_map)
        score_map_vis.save(os.path.join(score_renders_path, '{0:05d}'.format(idx) + "_score_vis.png"))
        
        gt_score_map_vis = one_channel_vis(gt_score_map)
        gt_score_map_vis.save(os.path.join(gt_score_map_path, '{0:05d}'.format(idx) + "_score_vis.png"))
        
        # save feature map
        score_map = score_map.cpu().numpy().astype(np.float16)
        torch.save(torch.tensor(score_map).half(), os.path.join(score_tensors_path, '{0:05d}'.format(idx) + "_smap.pt"))
        #############
        deltas = []
        # for _ in range(2):
            # deltas.append([
            #     random.uniform(0, 15),       # 1st element: random float in range 0-20
            #     random.uniform(0, 15),       # 2nd element: random float in range 0-20
            #     0,                           # 3rd element: 0
            #     random.uniform(0, 0.12),     # 4th element: random float in range 0-0.15
            #     random.uniform(0, 0.12),     # 5th element: random float in range 0-0.15
            #     0
            # ])
        deltas.append([20,10,0,0.1,0,0.05])
        deltas.append([-20,-8,0,0,0.05,0])
        render_novel_views(idx, view, gaussians, pipe_param, background, novel_paths, deltas[0], num=1)
        render_novel_views(idx, view, gaussians, pipe_param, background, novel_paths, deltas[1], num=2)






def render_sets(scene:Scene, gaussians:GaussianModel, model_param : ModelParams, 
                pipe_param:PipelineParams):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        render_test_pairs(model_param.model_path, "rendering/test_pairs", scene.loaded_iter, 
                        scene.getTestCameras(), gaussians, pipe_param, background)




# bash zenith_scripts/match_blending.sh
if __name__ == "__main__":
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model_param = ModelParams(parser, sentinel=True)
    pipe_param = PipelineParams(parser)
    args = get_combined_args(parser)
    args.iteration = 8000
    args.eval = True

    dataset = model_param.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    

    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    render_sets(scene, gaussians, model_param.extract(args), pipe_param.extract(args))


