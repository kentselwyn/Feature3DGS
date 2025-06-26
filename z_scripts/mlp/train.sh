dataset="pgt_7scenes_stairs"
path="/home/koki/code/cc/feature_3dgs_2/data/vis_loc/gsplatloc/7_scenes/$dataset"
out_path="/home/koki/code/cc/feature_3dgs_2/data/mlpckpt/7scenes/$dataset"

python -u -m mlp.train --path $path --out_path $out_path


# ( bash z_scripts/mlp/train.sh )
