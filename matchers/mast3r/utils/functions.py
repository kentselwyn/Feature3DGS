import numpy as np
import math
import cv2

def perform_rodrigues_transformation(rvec):
    try:
        R, _ = cv2.Rodrigues(rvec)
        return R
    except cv2.error as e:
        return False
    
def getPredictPos(img_name,pred_file):
    text = []
    with open(pred_file) as f:
        for line in f.readlines():
            data = line.split('/t/n')
            for str in data:
                sub_str = str.split(' ')
                text.append(sub_str)
    num_database = len(text)
    #print(text)
    for i in range(num_database):
        if img_name in text[i]:
            trans_predict = [0,0,0]
            ori_predict = [0,0,0,0]
            for j in range(4):
                ori_predict[j] = float(text[i][j+1])
            for k in range(3):
                trans_predict[k] = float(text[i][k+5])
    trans_predict = np.array(trans_predict)
    ori_predict = np.array(ori_predict)
    Rot = qvec2rotmat(ori_predict)
    w2c_predict = np.eye(4)
    w2c_predict[:3,:3] = Rot
    w2c_predict[:3,3] = trans_predict
    return w2c_predict

def cal_rot_error(cur_R,obs_R):
    r_err = np.matmul(cur_R, np.transpose(obs_R))
    r_err = cv2.Rodrigues(r_err)[0]
    # Extract the angle.
    r_err = np.linalg.norm(r_err) * 180 / math.pi
    return r_err

def cal_tran_error(cur_T,obs_T):
    return np.linalg.norm(cur_T-obs_T)* 100

#Calculate the rotation and translation errors between two camera poses.
def cal_campose_error(cur_pose_c2w,obs_pose_c2w):
    rot_error=cal_rot_error(cur_pose_c2w[:3,:3],obs_pose_c2w[:3,:3])
    translation_error = cal_tran_error(cur_pose_c2w[:3,3],obs_pose_c2w[:3,3])
    return rot_error,translation_error
    
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def getScale(depth_mast3r,depth_gs):
    depth_map1 = depth_mast3r
    depth_map2 = depth_gs
    valid_mask = (depth_map1 > 1e-1) & (depth_map2 > 1e-1)
    # 获取有效的深度值
    valid_depth1 = depth_map1[valid_mask]
    valid_depth2 = depth_map2[valid_mask]
    # 计算比例因子
    scale_factors = valid_depth2 / valid_depth1
    # 去掉极端值（例如，使用中位数来计算比例因子）
    if len(scale_factors) == 0:
        scale_factor = 1
    else:
        scale_factor = np.median(scale_factors)
    return scale_factor
