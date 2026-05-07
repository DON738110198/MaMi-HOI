import os
import numpy as np
import torch


def save_human_jnts_to_npz(text, human_jnts_list, dest_res_for_eval_npz_folder):
    curr_pred_global_jpos = human_jnts_list[0].detach().cpu().numpy()
    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, text + ".npz")
    np.savez(curr_seq_dest_res_npz_path, seq_name=text, \
            global_jpos=curr_pred_global_jpos) # T X 24 X 3 

def prep_res_folders(save_res_folder):
    res_root_folder = save_res_folder 
    # Prepare folder for saving npz results 
    dest_res_for_eval_npz_folder = os.path.join(res_root_folder, "res_npz_files")
    # Prepare folder for evaluation metrics 
    dest_metric_root_folder = os.path.join(res_root_folder, "evaluation_metrics_json")
    # Prepare folder for visualization 
    dest_out_vis_root_folder = os.path.join(res_root_folder, "single_window_cmp_settings")
    # Prepare folder for saving .obj files 
    dest_out_obj_root_folder = os.path.join(res_root_folder, "objs_single_window_cmp_settings")


    # Prepare folder for saving text json files 
    dest_out_text_json_folder = os.path.join(dest_out_vis_root_folder, "text_json_files")


    dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois_wo_guidance")
    dest_metric_folder = os.path.join(dest_metric_root_folder, "chois_wo_guidance") 
    dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois_wo_guidance") 
    dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois_wo_guidance")
 
    # Create folders 
    if not os.path.exists(dest_metric_folder):
        os.makedirs(dest_metric_folder) 
    if not os.path.exists(dest_out_vis_folder):
        os.makedirs(dest_out_vis_folder) 
    if not os.path.exists(dest_res_for_eval_npz_folder):
        os.makedirs(dest_res_for_eval_npz_folder)
    if not os.path.exists(dest_out_obj_folder):
        os.makedirs(dest_out_obj_folder) 
    if not os.path.exists(dest_out_text_json_folder):
        os.makedirs(dest_out_text_json_folder)

    dest_out_gt_vis_folder = os.path.join(dest_out_vis_root_folder, "0_gt")
    if not os.path.exists(dest_out_gt_vis_folder):
        os.makedirs(dest_out_gt_vis_folder) 

    return dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
        dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder

def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def compute_metrics(ori_verts_gt, ori_verts_pred, ori_jpos_gt, ori_jpos_pred, human_faces, \
    gt_trans, pred_trans, gt_rot_mat, pred_rot_mat, gt_obj_com_pos, pred_obj_com_pos, \
    gt_obj_rot_mat, pred_obj_rot_mat, gt_obj_verts, pred_obj_verts, obj_faces, \
    actual_len, use_joints24=True):
    # verts_gt: T X Nv X 3 
    # jpos_gt: T X J X 3
    # gt_trans: T X 3
    # gt_rot_mat: T X 22 X 3 X 3 
    # gt_obj_com_pos: T X 3
    # gt_obj_rot_mat: T X 3 X 3
    # human_faces: Nf X 3, array  
    # obj_verts: T X No X 3
    # obj_faces: Nf X 3, array  
    # gt_contact_label: T X 2 (left palm, right palm)
    # pred_contact_label: T X 2
    # actual_len: scale value 

    # ori_verts_gt = ori_verts_gt[:actual_len]
    # ori_verts_pred = ori_verts_pred[:actual_len]
    # ori_jpos_gt = ori_jpos_gt[:actual_len]
    # ori_jpos_pred = ori_jpos_pred[:actual_len]
    # gt_trans = gt_trans[:actual_len]
    # pred_trans = pred_trans[:actual_len]
    # gt_rot_mat = gt_rot_mat[:actual_len]
    # pred_rot_mat = pred_rot_mat[:actual_len]
    # gt_obj_com_pos = gt_obj_com_pos[:actual_len]
    # pred_obj_com_pos = pred_obj_com_pos[:actual_len] 
    # gt_obj_rot_mat = gt_obj_rot_mat[:actual_len]
    # pred_obj_rot_mat = pred_obj_rot_mat[:actual_len] 
    # gt_obj_verts = gt_obj_verts[:actual_len]
    # pred_obj_verts = pred_obj_verts[:actual_len]

    # Calculate global hand joint position error 
    if use_joints24:
        lhand_idx = 22 
        rhand_idx = 23 
    else:
        lhand_idx = 20
        rhand_idx = 21
    lhand_jpos_pred = ori_jpos_pred[:, lhand_idx, :].detach().cpu().numpy() 
    rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpos_gt = ori_jpos_gt[:, lhand_idx, :].detach().cpu().numpy()
    rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=1).mean() * 1000
    rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=1).mean() * 1000
    hand_jpe = (lhand_jpe+rhand_jpe)/2.0 

    # Calculate MPVPE  
    verts_pred = ori_verts_pred - ori_jpos_pred[:, 0:1]
    verts_gt = ori_verts_gt - ori_jpos_gt[:, 0:1]
    verts_pred = verts_pred.detach().cpu().numpy()
    verts_gt = verts_gt.detach().cpu().numpy()
    mpvpe = np.linalg.norm(verts_pred - verts_gt, axis=2).mean() * 1000

    # Calculate MPJPE 
    jpos_pred = ori_jpos_pred - ori_jpos_pred[:, 0:1] # zero out root
    jpos_gt = ori_jpos_gt - ori_jpos_gt[:, 0:1] 
    jpos_pred = jpos_pred.detach().cpu().numpy()
    jpos_gt = jpos_gt.detach().cpu().numpy()
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    # Caculate translation error 
    trans_err = np.linalg.norm(pred_trans.squeeze(0).detach().cpu().numpy() - gt_trans.squeeze(0).detach().cpu().numpy(), axis=1).mean() * 1000
    
    # Calculate rotation error
    rot_mat_pred = pred_rot_mat.detach().cpu().numpy()[:, 0] # Only evaluate for root rotation 
    rot_mat_gt = gt_rot_mat.detach().cpu().numpy()[:, 0]
    rot_dist = get_frobenious_norm_rot_only(rot_mat_pred.reshape(-1, 3, 3), rot_mat_gt.reshape(-1, 3, 3))
    # rot_dist = 0 

    # # Calculate foot sliding
    # floor_height = determine_floor_height_and_contacts(ori_jpos_pred.detach().cpu().numpy(), fps=30)
    # gt_floor_height = determine_floor_height_and_contacts(ori_jpos_gt.detach().cpu().numpy(), fps=30)
    # # print("floor height:{0}".format(floor_height)) 
    # # print("gt floor height:{0}".format(gt_floor_height)) 

    # foot_sliding_jnts = compute_foot_sliding_for_smpl(ori_jpos_pred.detach().cpu().numpy(), floor_height)
    # gt_foot_sliding_jnts = compute_foot_sliding_for_smpl(ori_jpos_gt.detach().cpu().numpy(), gt_floor_height)

    # Compute contact score 
    num_obj_verts = gt_obj_verts.shape[1]
    if use_joints24:
        # contact_threh = 0.05
        contact_threh = 0.05
    else:
        contact_threh = 0.10 

    gt_lhand_jnt = ori_jpos_gt[:, lhand_idx, :] # T X 3 
    gt_rhand_jnt = ori_jpos_gt[:, rhand_idx, :] # T X 3 

    # What if the joint is in the object? already penetrate? 
    gt_lhand2obj_dist = torch.sqrt(((gt_lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    gt_rhand2obj_dist = torch.sqrt(((gt_rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - gt_obj_verts.to(gt_rhand_jnt.device))**2).sum(dim=-1)) # T X N  

    gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0] # T 
    gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0] # T 

    gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threh)
    gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)

    lhand_jnt = ori_jpos_pred[:, lhand_idx, :] # T X 3 
    rhand_jnt = ori_jpos_pred[:, rhand_idx, :] # T X 3 

    lhand2obj_dist = torch.sqrt(((lhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(lhand_jnt.device))**2).sum(dim=-1)) # T X N  
    rhand2obj_dist = torch.sqrt(((rhand_jnt[:, None, :].repeat(1, num_obj_verts, 1) - pred_obj_verts.to(rhand_jnt.device))**2).sum(dim=-1)) # T X N  
   
    lhand2obj_dist_min = lhand2obj_dist.min(dim=1)[0] # T 
    rhand2obj_dist_min = rhand2obj_dist.min(dim=1)[0] # T 

    lhand_contact = (lhand2obj_dist_min < contact_threh)
    rhand_contact = (rhand2obj_dist_min < contact_threh)

    num_steps = gt_lhand_contact.shape[0]

    # Compute the distance between hand joint and object for frames that are in contact with object in GT. 
    contact_dist = 0
    gt_contact_dist = 0 

    gt_contact_cnt = 0
    for idx in range(num_steps):
        if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
            gt_contact_cnt += 1 

            contact_dist += min(lhand2obj_dist_min[idx], rhand2obj_dist_min[idx])
            gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx])

    if gt_contact_cnt == 0:
        contact_dist = 0 
        gt_contact_dist = 0 
    else:
        contact_dist = contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)
        gt_contact_dist = gt_contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)

    pred_contact_cnt = 0

    # Compute precision and recall for contact. 
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(num_steps):
        gt_in_contact = (gt_lhand_contact[idx] or gt_rhand_contact[idx]) 
        pred_in_contact = (lhand_contact[idx] or rhand_contact[idx])
        if gt_in_contact and pred_in_contact:
            TP += 1

        if (not gt_in_contact) and pred_in_contact:
            FP += 1

        if (not gt_in_contact) and (not pred_in_contact):
            TN += 1

        if gt_in_contact and (not pred_in_contact):
            FN += 1

        if pred_in_contact:
            pred_contact_cnt += 1 

    gt_contact_percent = gt_contact_cnt /float(num_steps)
    pred_contact_percent = pred_contact_cnt / float(num_steps) 

    contact_acc = (TP+TN)/(TP+FP+TN+FN)

    if (TP+FP) == 0: # Prediction no contact!!!
        contact_precision = 0
        print("Contact precision, TP + FP == 0!!")
    else:
        contact_precision = TP/(TP+FP)
    
    if (TP+FN) == 0: # GT no contact! 
        contact_recall = 0
        print("Contact recall, TP + FN == 0!!")
    else:
        contact_recall = TP/(TP+FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0 
    else:
        contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall) 
   
    # Compute object rotation error.
    obj_rot_mat_pred = pred_obj_rot_mat.detach().cpu().numpy() 
    obj_rot_mat_gt = gt_obj_rot_mat.detach().cpu().numpy()
    obj_rot_dist = get_frobenious_norm_rot_only(obj_rot_mat_pred.reshape(-1, 3, 3), obj_rot_mat_gt.reshape(-1, 3, 3))
    # obj_rot_dist = 0 

    # Compute com error. 
    obj_com_pos_err = np.linalg.norm(pred_obj_com_pos.detach().cpu().numpy() - gt_obj_com_pos.detach().cpu().numpy(), axis=1).mean() * 1000

    # # Compute matching between the prediction and input conditions. 
    # start_obj_com_pos_err = np.linalg.norm(pred_obj_com_pos[0:1].detach().cpu().numpy() - gt_obj_com_pos[0:1].detach().cpu().numpy(), axis=1).mean() * 1000
    # end_obj_com_pos_err = np.linalg.norm(pred_obj_com_pos[-1:].detach().cpu().numpy() - gt_obj_com_pos[-1:].detach().cpu().numpy(), axis=1).mean() * 1000

    # waypoints_index_list = [29, 59, 89] 
    # waypoints_xy_pos_err = np.linalg.norm(pred_obj_com_pos[waypoints_index_list, :2].detach().cpu().numpy() - \
    #                 gt_obj_com_pos[waypoints_index_list, :2].detach().cpu().numpy(), axis=1).mean() * 1000

    return lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, gt_contact_percent, pred_contact_percent, \
    contact_precision, contact_recall, contact_acc, contact_f1_score, \
    obj_rot_dist, obj_com_pos_err
