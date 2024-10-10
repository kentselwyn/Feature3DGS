from scene.gaussian_model import GaussianModel



def load_gaussian(path = "/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_00/A/outputs/outsp/point_cloud/iteration_7000/point_cloud.ply"):
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(path)

    print(gaussians.get_opacity.shape)

    breakpoint()



# python -m codes.used_codes.vis_gaussian
if __name__=="__main__":
    # load_gaussian("/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_00/A/outputs/6/point_cloud/iteration_7000/point_cloud.ply")
    load_gaussian("/home/koki/code/cc/feature_3dgs_2/all_data/scene0000_00/A/outputs/output/point_cloud/iteration_10000/point_cloud.ply")


