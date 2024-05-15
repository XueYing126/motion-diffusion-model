'''
For evaluation of 263D on smpl evaluator
Sample for all test texts and transform them to smpl
Input folder: './HumanML3D/new_joint_vecs/'
Output folder: './dataset/HumanML3D/eval/sample_0_/'

run script: python -m eval.sample_test --model_path ./save/humanml_enc_512_50steps/model000750000.pt --device 1
'''
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from utils.parser_util import generate_args
from eval.smplify import Joints2SMPL
from data_loaders.tensors import collate
from sample.generate import load_dataset
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.scripts.motion_process import recover_from_ric

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = generate_args()
    max_frames = 196
    n_frames = 120

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(args.device)
    model.eval()  # disable random masking


    ###############################################
    # sample all test
    text_dir = './dataset/HumanML3D/texts'
    split_file = './dataset/HumanML3D/test.txt'
    tt = str(args.rep_t).zfill(2)
    save_folder = f'./dataset/HumanML3D/eval/sample_{tt}'
    
    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())
    random.shuffle(id_list)
    
    for name in tqdm(id_list):
        save_path = pjoin(save_folder, name + '.npy')
        if os.path.exists(save_path):
            continue
        
        try:
            motion_path = pjoin("./dataset/HumanML3D/new_joint_vecs", name + '.npy')
            ori_motion = np.load(motion_path)
            n_frames = ori_motion.shape[0]
        except:
            print("can not find ", motion_path)
            continue

        text_list = []
        with cs.open(pjoin(text_dir, name + '.txt')) as f:
            for line in f.readlines():
                line_split = line.strip().split('#')
                caption = line_split[0]
                text_list.append(caption)

        texts = [random.choice(text_list)]
        args.num_samples = len(texts)
        args.batch_size = args.num_samples

        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        
        _, model_kwargs = collate(collate_args)


        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=args.device) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        n_joints = 22
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float() #[bs, 1, 196, 263]
        joints = recover_from_ric(sample, n_joints)[0][0] #[bs, 1, 196, 22, 3]
        joints[...,[0, 2]] *=-1

        motion135d, trans, pose_6d = Joints2SMPL(joints, 50, args.device)#(bs, 135)
        
        save_data = {
            "bdata_trans": trans,
            "pose_6d":pose_6d, 
            "jtr": joints,
        }
        np.save(save_path, save_data)