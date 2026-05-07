import pytorch3d.transforms as transforms
import argparse
import os
import numpy as np
import yaml
import random
import json
import copy

import trimesh

from matplotlib import pyplot as plt
from pathlib import Path

import torch

import torch.nn.functional as F

import pytorch3d.transforms as transforms

from ema_pytorch import EMA
import sys
sys.path.append("../../")
sys.path.append("../")
from manip.data.cano_traj_dataset import  quat_ik_torch, quat_fk_torch

# from manip.model.transformer_object_motion_cond_diffusion import ObjectCondGaussianDiffusion

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

# from manip.lafan1.utils import quat_inv, quat_mul, quat_between, normalize, quat_normalize

# from t2m_eval.evaluation_metrics import compute_metrics, determine_floor_height_and_contacts, compute_metrics_long_seq
import argparse
import os
import sys

from visualize import vis_utils
import shutil
from tqdm import tqdm

from scipy.spatial.transform import Rotation
import random
torch.manual_seed(1)
random.seed(1)

from visualize.vis_utils import simplified_mesh
def gen_vis_res_generic(motion, motion_obj, params, vis_gt=True, vis_tag=None, dest_out_vid_path=None, dest_mesh_vis_folder=None, save_obj_only=False):

        # Prepare list used for evaluation.
        human_jnts_list = []
        human_verts_list = []
        obj_verts_list = []
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = []

        human_joints = motion["motion"] # (25, 6, 196)
        human_verts = motion["vertices"] # (6890, 3, 196)
        human_mesh_faces = motion["faces"] # (13776, 3)
        obj_verts = motion_obj["vertices"] # (505, 3, 196)
        obj_mesh_faces = motion_obj["faces"] # (999, 3)
        human_trans = motion["root_translation"] # (3, 196)

        human_jnts_list.append(human_joints)
        human_verts_list.append(human_verts)
        obj_verts_list.append(obj_verts)
        trans_list.append(human_trans)
        human_mesh_faces_list.append(human_mesh_faces)
        obj_mesh_faces_list.append(obj_mesh_faces)
        # all_res_list: N X T X (3+9)
        # num_seq = all_res_list.shape[0]


        if dest_mesh_vis_folder is None:
            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(params.processed_path, "blender_mesh_vis")
            else:
                dest_mesh_vis_folder = os.path.join(params.processed_path, vis_tag)

        if not os.path.exists(dest_mesh_vis_folder):
            os.makedirs(dest_mesh_vis_folder)

        if vis_gt:
            ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                            "ball_objs_step_"+"_bs_idx_"+"_gt")
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                            "objs_step_"+"_bs_idx_"+"_gt")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                            "imgs_step_"+"_bs_idx_"+"_gt")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            "vid_step_"+"_bs_idx_"+"_gt.mp4")



        # For faster debug visualization!!
        # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
        # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3

        save_verts_faces_to_mesh_file_w_object(np.transpose(human_verts, (2, 0, 1)), \
                human_mesh_faces, \
                np.transpose(obj_verts, (2, 0, 1)), obj_mesh_faces, mesh_save_folder)


        floor_blend_path = os.path.join("/data2/wh/hoi_diffusion_model/processed_data", "blender_files/floor_colorful_mat.blend")

        is_compute_metric = True
        if is_compute_metric:
            return human_verts_list, human_jnts_list, trans_list, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path

        if dest_out_vid_path is None:
            dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")
        if not os.path.exists(dest_out_vid_path):
            if not vis_gt: # Skip GT visualiation
                if not save_obj_only:
                    run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                            condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                            scene_blend_path=floor_blend_path)

        if vis_gt: # here
            if not save_obj_only:
                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                        condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=False, \
                        scene_blend_path=floor_blend_path)


        # return human_verts_list, human_jnts_list, trans_list, global_rot_mat, pred_seq_com_pos, pred_obj_rot_mat, \
        # obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path
        return human_verts_list, human_jnts_list, trans_list, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 zero

    B = smpl_trans.shape[0] # (BS*T)

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue

        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)

        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3
        lmiddle_index= 28
        rmiddle_index = 43
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1)
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]

    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3


    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3

    mesh_faces = pred_body.f

    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces

def process_save_motion(input_path, motion, motion_obj, obj_name, params):


    # vertices_list = []
    # faces_list = []
    # for b in range(motion.shape[0]):
    #     tmp_obj_name = obj_name[b].split("_")[2]
    #     mesh_path = os.path.join(params.obj_mesh_path, simplified_mesh[tmp_obj_name])
    #     tmp_mesh = trimesh.load(mesh_path)
    #     vertices = tmp_mesh.vertices
    #     faces = tmp_mesh.faces
    #     center = np.mean(vertices, 0)
    #     vertices -= center
    #     angle, trans = motion_obj[b, 0, :3], motion_obj[b, 0, 3:]
    #     rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
    #     vertices = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
    #     vertices = vertices.transpose(1, 2, 0)
    #     vertices_list.append(vertices)
    #     faces_list.append(faces)


    npy2obj_object = vis_utils.npy2obj_object(input_path, params.obj_mesh_path, 0, 0,
                                       device=params.device, cuda=params.cuda, if_color=True)

    # # human
    npy2obj = vis_utils.npy2obj(input_path, 0, 0,
                                device=params.device, cuda=params.cuda, if_color=True)
    out_npy_path = input_path.replace(".npy", "smpl_param.npy")
    out_obj_npy_path = input_path.replace(".npy", "obj_param.npy")
    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
    npy2obj_object.save_npy(out_obj_npy_path)

def read_processed_motion(params):
    motion_path = os.path.join(params.processed_path, "gt_resultssmpl_param.npy")
    motion_obj_path = os.path.join(params.processed_path, "gt_resultsobj_param.npy")
    motion = np.load(motion_path, allow_pickle=True)
    motion_obj = np.load(motion_obj_path, allow_pickle=True)
    return motion.item(), motion_obj.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default = "/data2/wh/hoi_diffusion_model/save/my_hoi_diff_enc_step300000_resume_4/gt_results.npy", help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--processed_path", type=str, default="/data2/wh/hoi_diffusion_model/save/my_hoi_diff_enc_step300000_resume_4", help='')
    parser.add_argument("--obj_mesh_path", type=str, default='/data2/wh/hoi_diffusion_model/dataset/behave_t2m/object_mesh')
    params = parser.parse_args()

    input_path = params.input_path
    data_dict = np.load(input_path, allow_pickle=True)
    motion = data_dict.item()["motion"] # (10, 22, 3, 196)
    motion_obj = data_dict.item()["motion_obj"] # (10, 1, 6, 196)
    obj_name = data_dict.item()["obj_name"]

    # process_save_motion(params.input_path, motion, motion_obj, obj_name, params)
    motion, motion_obj = read_processed_motion(params)
    gen_vis_res_generic(motion, motion_obj, params=params)

