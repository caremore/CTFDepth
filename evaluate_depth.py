import os

os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import readlines
from options import MonodepthOptions
import datasets, networks
from networks.layers import transformation_from_parameters, disp_to_depth
import tqdm
import torch.nn.functional as F

cv2.setNumThreads(0)
STEREO_SCALE_FACTOR = 5.4
import matplotlib.pyplot as plt
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
#python -m dynamicdepth.evaluate_depth --load_weights_folder ./pre-trained/ --eval_mono
#abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
#&   0.270  &   0.449  &   1.538  &   0.385  &   0.500  &   0.801  &   0.909  \\

#python -m dynamicdepth.evaluate_depth --load_weights_folder ./pre-trained/ --eval_mono --zero_cost_volume --static_camera
# abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
#&   0.312  &   1.830  &   1.168  &   0.315  &   0.615  &   0.864  &   0.948  \\

import os

ADDRESS = "./logs/fsea/imprseaErra192_1/weights_49"#usod10k,fsea
# The Address where you want to save generated predicted result.

if not os.path.exists(ADDRESS):
    os.makedirs(ADDRESS, exist_ok=True)
if not os.path.exists(ADDRESS + '/mono_img_mg'):
    os.mkdir(ADDRESS + '/mono_img_mg')

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)
    print('=========== Frame to be used ==========', frames_to_load)
    # Just using the past one frame for evaluation.
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    if opt.ext_disp_to_eval is None:
        #opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)
        print("-> Loading weights from {}".format(opt.load_weights_folder))
        # # Setup dataloaders
        #
        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:

            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_class = networks.ResnetEncoderMatching

        encoder_dict = torch.load(encoder_path)
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        filenames = readlines(os.path.join(opt.data_path, "tst.txt"))
        if opt.eval_split == 'fsea':
            dataset = datasets.FSEADepthDataset(opt.data_path, filenames, HEIGHT, WIDTH,
                                      frames_to_load, 4, opt.use_depth_hints,
                                      opt.depth_hint_path,
                                      is_train=False, img_ext='.png',is_test=True)
        else:
            dataset = datasets.USOD10kDepthDataset("/root/autodl-fs/pythondata/USOD10k/TE",
                                         HEIGHT, WIDTH,
                                         img_ext='.png')

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        # setup models
        if opt.eval_teacher:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False)
        else:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins)
            pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"), weights_only=True)
            pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"), weights_only=True)
            pose_enc = networks.PoseResnetEncoder(18, False, num_input_images=2)

            pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                            num_frames_to_predict_for=2)

            pose_enc.load_state_dict(pose_enc_dict, strict=True)
            pose_dec.load_state_dict(pose_dec_dict, strict=True)

            min_depth_bin = encoder_dict.get('min_depth_bin')
            max_depth_bin = encoder_dict.get('max_depth_bin')

            pose_enc.eval()
            pose_dec.eval()

            if torch.cuda.is_available():
                pose_enc.cuda()
                pose_dec.cuda()
        if opt.use_simmim:
            simmim_path = os.path.join(opt.load_weights_folder, "simmim.pth")
            simmim = networks.SimmimMode
            simmim.load_state_dict(torch.load(simmim_path))
        # ------------ Loading Encoder ------------
        encoder = encoder_class(**encoder_opts)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        depth_decoder.eval()
        simmim.eval()

        if torch.cuda.is_available():
            encoder.cuda()
            depth_decoder.cuda()
            simmim.cuda()
        pred_disps = []
        gt_depths = []
        folder_lst = []
        frame_index_lst = []
        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color_aug', 0, 0)]
                #input_color = data[('color_aug', 0, 0)]
                #input_color_enhanced = data[('color', 0, 0)]
                input_depth = (data['depth_gt'])[:, 0].numpy()
                folder = data["folder"][0]
                frame_index = data["frame_index"][0]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                if opt.eval_teacher:
                    # -------------------------simin_model------------------------------------------
                    if opt.use_simmim:
                        _, _, masknet_featlst = simmim(input_color, input_color, None)
                    # -------------------------simin_model------------------------------------------
                    output = encoder(input_color,masknet_featlst)
                    output = depth_decoder(output)
                else:
                    if opt.static_camera:
                        for f_i in frames_to_load:
                            data["color_aug", f_i, 0] = data[('color_aug', 0, 0)]
                    # predict poses
                    pose_feats = {f_i: data["color_aug", f_i, 0] for f_i in frames_to_load}
                    if torch.cuda.is_available():
                        pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in frames_to_load[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)
                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])
                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False)
                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, data[('relative_pose', fi - 1)])
                        data[('relative_pose', fi)] = pose
                    lookup_frames = [data[('color_aug', idx, 0)] for idx in frames_to_load[1:]]
                    lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w
                    relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                    relative_poses = torch.stack(relative_poses, 1)
                    K = data[('K', 2)]  # quarter resolution for matching
                    invK = data[('inv_K', 2)]
                    if torch.cuda.is_available():
                        lookup_frames = lookup_frames.cuda()
                        relative_poses = relative_poses.cuda()
                        K = K.cuda()
                        invK = invK.cuda()
                    if opt.zero_cost_volume:
                        relative_poses *= 0
                    if opt.post_process:
                        raise NotImplementedError

                    # -------------------------simin_model------------------------------------------
                    if opt.use_simmim:
                        _, _, masknet_featlst = simmim(input_color, input_color.cuda(), None)
                    # -------------------------simin_model------------------------------------------
                    # ------------ Finished Coarse ------------
                    output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                           relative_poses,
                                                           K,
                                                           invK,
                                                           masknet_featlst,
                                                           min_depth_bin, max_depth_bin)
                    output = depth_decoder(output)
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                gt_depths.append(input_depth)
                folder_lst.append(folder)
                frame_index_lst.append(frame_index)

                """
                save_path = '/mnt/nas/kaichen/eng/TTC/many/02_visual/gt_img/%s_rgb.png'%i
                print(input_color.size(), '=============', i)
                cv2.imwrite(save_path, input_color[0].permute(1,2,0).cpu().numpy()[:,:,::-1]*255.)
                """
        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)
        #folder_lst = np.concatenate(folder_lst)
        #frame_index_lst = np.concatenate(frame_index_lst)
        print('finished predicting!')
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        if opt.zero_cost_volume:
            tag = "zero_cv"
        elif opt.eval_teacher:
            tag = "teacher"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_split.npy".format(tag, opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()
    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)
        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    for index in range(pred_disps.shape[0]):
        depth_path = ADDRESS + '/mono_img_mg/' +f"{folder_lst[index]}_{frame_index_lst[index]}_depth.png"

        if opt.eval_split == 'fsea':
            depth_resized = cv2.resize(
                pred_disps[index], ( 968,608), interpolation=cv2.INTER_CUBIC
            )
        else:
            depth_resized = cv2.resize(
                pred_disps[index], ( 640,480), interpolation=cv2.INTER_CUBIC
            )


        vmax = np.percentile(depth_resized, 95)
        vmin = depth_resized.min()
        plt.imsave(depth_path, depth_resized, cmap='magma',vmin = vmin, vmax=vmax,dpi=300)#magma,plasma
        plt.axis('off')

    print("-> Evaluating")
    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True

        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # if opt.eval_split == 'fsea':
        #     mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        # else:
        #     mask = gt_depth > 0
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        error_map = np.zeros(pred_depth.shape)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))
        error_map[mask] = np.abs(pred_depth - gt_depth) / gt_depth
        depth_path = os.path.join(ADDRESS,f"{folder_lst[i]}_{frame_index_lst[i]}_depth.png" )#'%s_depth.png' % i
        vmax = np.percentile(pred_depth, 1)
        plt.imsave(depth_path, error_map, cmap='Reds', vmax=1)
        plt.axis('off')

    if opt.save_pred_disps:
        print("saving errors")
        if opt.zero_cost_volume:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())


