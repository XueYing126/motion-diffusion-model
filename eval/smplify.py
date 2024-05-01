'''
credit to joints2smpl
https://github.com/wangsen1312/joints2smpl

Use SMPLify to process humanact12 dataset and obtain SMPL parameters

Input folder: './pose_data/humanact12'
Output folder: './humanact12/
'''

import os
import codecs as cs
import numpy as np
from tqdm import tqdm
import torch
from joints2smpl.src import config
import smplx
import h5py
from joints2smpl.src.smplify import SMPLify3D
import random
from utils import utils_transform
from os.path import join as pjoin
from data_loaders.humanml.scripts.motion_process import recover_from_ric

def Joints2SMPL(input_joints, num_smplify_iters = 50):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input_joints = input_joints[:, :, [0, 2, 1]] # amass stands on x, y

    batch_size = input_joints.shape[0] # #[seq_len, 22, 3]

    smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                            model_type="smpl", gender="neutral", ext="pkl",
                            batch_size=batch_size).to(device)

    # ## --- load the mean pose as original ----
    smpl_mean_file = config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

    # # #-------------initialize SMPLify
    # num_smplify_iters = 20 # 150, 100
    smplify = SMPLify3D(smplxmodel=smplmodel,
                        batch_size=batch_size,
                        joints_category="AMASS",
                        num_iters=num_smplify_iters,
                        device=device)


    keypoints_3d = torch.Tensor(input_joints).to(device).float()

    pred_betas = init_mean_shape
    pred_pose = init_mean_pose
    pred_cam_t = cam_trans_zero

    confidence_input = torch.ones(22)
        

    new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
    new_opt_cam_t, new_opt_joint_loss = smplify(
        pred_pose.detach(),
        pred_betas.detach(),
        pred_cam_t.detach(),
        keypoints_3d,
        conf_3d=confidence_input.to(device),
        # seq_ind=idx
    )
    # print(new_opt_joint_loss)

    poses = new_opt_pose.detach().cpu().numpy()[:, :3*22] #[199, 72]

    betas = new_opt_betas.mean(axis=0).detach().cpu().numpy()
    trans = keypoints_3d[:, 0].detach().cpu().numpy()
    
    pose_aa = torch.Tensor(poses).reshape(-1,3)
    pose_6d = utils_transform.aa2sixd(pose_aa).reshape(poses.shape[0],-1).numpy()
    
    motion = np.concatenate((trans, pose_6d[..., :6*22]), axis=1)
   
    return motion


if __name__ == "__main__":

    split_file = './dataset/HumanML3D/test.txt'
    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())

    random.shuffle(id_list)
    for name in tqdm(id_list):
        save_path = pjoin("./dataset/HumanML3D/eval/gt", name + '.npy')

        if os.path.exists(save_path):
            print(f'{save_path} already exists')
            continue

        motion = np.load(pjoin("./dataset/HumanML3D/new_joint_vecs", name + '.npy')) #(seq_len, 263)
        joints = recover_from_ric(torch.tensor(motion), 22)#[seq_len, 22, 3]
        motion153d = Joints2SMPL(joints, 50)
        
        np.save(save_path, motion153d)
        