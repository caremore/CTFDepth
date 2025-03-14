import os
import torch
import networks
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import readlines, sec_to_hm_str
import torch.nn.functional as F
import datasets
from networks.layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors
import numpy as np
import random
import json
import time
import logging
import cv2
import matplotlib.pyplot as plt


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class Trainer:
    def __init__(self,options):

        logging.info('--------------------Mynew--------------------')
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        # ====================================================================
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.train_teacher_and_pose = not self.opt.freeze_teacher_and_pose
        if self.train_teacher_and_pose:
            logging.info('using adaptive depth binning!')
            self.min_depth_tracker = 0.1
            self.max_depth_tracker = 10.0
        else:
            logging.info('fixing pose network and monocular network!')

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = [0]
        if self.opt.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        logging.info('Loading frames: {}'.format(frames_to_load))
        logging.info('Matched Frames: {}'.format(self.matching_ids))  # Only using -1 frame to construct the cost volume.
        # ====================================================================
        self.models["encoder"] = networks.ResnetEncoderMatching(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            input_height=self.opt.height, input_width=self.opt.width,
            adaptive_bins=True, min_depth_bin=0.1, max_depth_bin=20.0,
            depth_binning=self.opt.depth_binning, num_depth_bins=self.opt.num_depth_bins)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["mono_encoder"] = \
            networks.ResnetEncoder(18, self.opt.weights_init == "pretrained")
        self.models["mono_encoder"].to(self.device)

        self.models["mono_depth"] = \
            networks.DepthDecoder(self.models["mono_encoder"].num_ch_enc, self.opt.scales)
        self.models["mono_depth"].to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["mono_encoder"].parameters())
            self.parameters_to_train += list(self.models["mono_depth"].parameters())

        self.models["pose_encoder"] = \
            networks.PoseResnetEncoder(18, self.opt.weights_init == "pretrained",
                                   num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = \
            networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                 num_input_features=1,
                                 num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)

        if self.opt.use_simmim:
            logging.info(f"Training use simmimmodel ")
            self.models["simmim"] = networks.SimmimMode
            self.models["simmim"].to(self.device)
            self.parameters_to_train += list(self.models["simmim"].parameters())
        #-------------------------------------------------------------------------------------
        total_params = []
        total_params.append(sum(p.numel() for p in self.models["encoder"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["depth"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["mono_encoder"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["mono_depth"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["pose_encoder"].parameters()))
        total_params.append(sum(p.numel() for p in self.models["pose"].parameters()))
        if self.opt.use_simmim:
            total_params.append(sum(p.numel() for p in self.models["simmim"].parameters()))
        total_params_weights = sum(total_params) / 1e6
        logging.info(f'--------------------weights: {total_params_weights}--------------------')
        #---------------------------------Memory ------Flops ------------------------------------

        # -------------------------------------------------------------------------------------

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
        self.model_optimizer, self.opt.scheduler_step_size, 0.1)


        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.mono_weights_folder is not None:
            self.load_mono_model()

        logging.info(f"Training model named:\n  {self.opt.model_name}")
        logging.info(f"Models and tensorboard events files are saved to:\n  {self.opt.log_dir}")
        logging.info(f"Training is using:\n  {self.device}")
        # ====================================================================
        fpath = os.path.join(self.opt.data_path, "{}.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        if self.opt.use_simmim:
            train_datasets = datasets.FSEADepthDataset_masked(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                                                    self.opt.frame_ids, 4, self.opt.use_depth_hints,
                                                    self.opt.depth_hint_path,
                                                    is_train=True, img_ext='.png',use_transform = True)
            val_datasets = datasets.FSEADepthDataset_masked(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                                                     self.opt.frame_ids, 4, self.opt.use_depth_hints,
                                                     self.opt.depth_hint_path,
                                                     is_train=False, img_ext='.png',use_transform = True)
        else:
            train_datasets = datasets.FSEADepthDataset(self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                                                    self.opt.frame_ids, 4, self.opt.use_depth_hints,
                                                    self.opt.depth_hint_path,
                                                    is_train=True, img_ext='.png')
            val_datasets = datasets.FSEADepthDataset(self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                                                    self.opt.frame_ids, 4, self.opt.use_depth_hints,
                                                    self.opt.depth_hint_path,
                                                    is_train=False, img_ext='.png')
        self.train_loader = DataLoader(train_datasets, batch_size=self.opt.batch_size, shuffle=True,
                                       num_workers=self.opt.num_workers,
                                       pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
        self.val_loader = DataLoader(val_datasets, batch_size=self.opt.batch_size, shuffle=False,
                                     num_workers=self.opt.num_workers,
                                     pin_memory=False, drop_last=True)
        self.val_iter = iter(self.val_loader)
        logging.info(f'{len(self.train_loader)}===============Length of train')
        logging.info(f'{len(self.val_loader)}===============Length of val')
        # ======================================================tst============================================================
        # fpath = os.path.join(self.opt.data_path, "{}.txt")
        # train_filenames = readlines(fpath.format("train"))
        # val_filenames = readlines(fpath.format("val"))
        # dataset = datasets.USOD10kDepthDataset("/root/autodl-fs/pythondata/USOD10k/TE", self.opts.img_h,
        #                                        self.opts.img_w,
        #                                        img_ext='.png')

        tst1_filenames = readlines(os.path.join(self.opt.data_path, "tst.txt"))
        if "fsea" in self.opt.data_path:
            frames_to_load = [0]
            if self.opt.use_future_frame:
                frames_to_load.append(1)
            for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
                if idx not in frames_to_load:
                    frames_to_load.append(idx)
            tst1_dataset = datasets.FSEADepthDataset(self.opt.data_path, tst1_filenames, self.opt.height, self.opt.width,
                                                 frames_to_load, 4, self.opt.use_depth_hints,
                                                 self.opt.depth_hint_path,
                                                 is_train=False, img_ext='.png')
            self.tst1_loader = DataLoader(
                tst1_dataset, self.opt.batch_size, False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
            logging.info(f'{len(self.tst1_loader)}===============Length of tst1')
            #self.tst1_iter = iter(self.tst1_loader)

        tst2_dataset = datasets.USOD10kDepthDataset("/root/autodl-fs/pythondata/USOD10k/TE",
                                                    self.opt.height, self.opt.width,
                                                    img_ext='.png')

        self.tst2_loader = DataLoader(
            tst2_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        logging.info(f'{len(self.tst2_loader)}===============Length of tst2')

        #self.tst2_iter = iter(self.tst2_loader)
        # ======================================================tst============================================================

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        logging.info("There are {:d} training items and {:d} validation items\n".format(
            len(train_datasets), len(val_datasets)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """

        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['depth', 'encoder']:
                    m.train()
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
    #===========================================================
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch == self.opt.freeze_teacher_epoch:
                self.freeze_teacher()

            self.run_epoch()
            self.test_epoch()
            self.test_epoch2()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
    # ===========================================================
    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            print('freezing teacher and pose networks!')

            # here we reinitialise our optimizer to ensure there are no updates to the
            # teacher and pose networks
            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, self.opt.scheduler_step_size, 0.1)

            # set eval so that teacher + pose batch norm is running average
            self.set_eval()
            # set train so that multi batch norm is in train mode
            self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        logging.info("============> Training{} <============".format(self.epoch))

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 500 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            if self.step == self.opt.freeze_teacher_step:
                self.freeze_teacher()

            self.step += 1
        self.model_lr_scheduler.step()
    def test_epoch(self):
        logging.info("============> Tst1 {} <============".format(self.epoch))
        self.set_eval()
        pred_disps = []
        gt_depths = []
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        HEIGHT, WIDTH = self.opt.height, self.opt.width
        logging.info("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        frames_to_load = [0]
        if self.opt.use_future_frame:
            frames_to_load.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            if idx not in frames_to_load:
                frames_to_load.append(idx)
        with (torch.no_grad()):
            for i, data in enumerate(self.tst1_loader):

                input_color = data[('color_aug', 0, 0)]
                input_color_enhanced = data[('color', 0, 0)]
                input_depth = (data['depth_gt'])[:, 0].numpy()

                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                if self.opt.eval_teacher:
                    # -------------------------simin_model------------------------------------------
                    if self.opt.use_simmim:
                        _, _, masknet_featlst = self.models['simmim'
                                        ](input_color,input_color_enhanced.to(self.device),None)
                    # -------------------------simin_model------------------------------------------
                    output = self.models["mono_encoder"](input_color,masknet_featlst)
                    output = self.models["mono_depth"](output)
                else:
                    if self.opt.static_camera:
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
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)
                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])
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
                    if self.opt.zero_cost_volume:
                        relative_poses *= 0
                    if self.opt.post_process:
                        raise NotImplementedError
                    min_depth_bin = self.min_depth_tracker
                    max_depth_bin = self.max_depth_tracker

                    # -------------------------simin_model----------------------------------------------------------------------------
                    if self.opt.use_simmim:
                        _, _, masknet_featlst = self.models['simmim'](input_color, input_color_enhanced.to(self.device), None)
                    # -------------------------simin_model----------------------------------------------------------------------------
                    output, lowest_cost, costvol = self.models["encoder"](input_color, lookup_frames,
                                                           relative_poses,
                                                           K,
                                                           invK,
                                                           masknet_featlst,
                                                           min_depth_bin, max_depth_bin)
                    output = self.models["depth"](output)
                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                gt_depths.append(input_depth)


        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)#(150, 128, 416)


        errors = []
        ratios = []
        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = np.squeeze(pred_disps[i])
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            # if self.opt.eval_split == 'fsea':
            #     mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            # else:
            #     mask = gt_depth > 0

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            #---------------------------------------------------
            # pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            # pred_depth[pred_depth > 80] = 80
            #-------------------------------------------------
            errors.append(compute_errors(gt_depth, pred_depth))
        mean_errors = np.array(errors).mean(0)
        logging.info("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        logging.info("\n-> Done!")
        self.set_train()

    def test_epoch2(self):
        logging.info("============> Tst2 {} <============".format(self.epoch))
        self.set_eval()
        pred_disps = []
        gt_depths = []
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        HEIGHT, WIDTH = self.opt.height, self.opt.width
        logging.info("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
        frames_to_load = [0]
        if self.opt.use_future_frame:
            frames_to_load.append(1)
        for idx in range(-1, -1 - self.opt.num_matching_frames, -1):
            if idx not in frames_to_load:
                frames_to_load.append(idx)
        with torch.no_grad():
            for i, data in enumerate(self.tst2_loader):

                input_color = data[('color_aug', 0, 0)]
                input_depth = (data['depth_gt'])[:, 0].numpy()

                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                if self.opt.eval_teacher:
                    # -------------------------simin_model------------------------------------------
                    if self.opt.use_simmim:
                        _, _, masknet_featlst = self.models['simmim'
                                        ](input_color,input_color.to(self.device),None)
                    # -------------------------simin_model------------------------------------------
                    output = self.models["mono_encoder"](input_color,masknet_featlst)
                    output = self.models["mono_depth"](output)
                else:

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
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                            axisangle, translation = self.models["pose"](pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)
                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])
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
                    #if self.opt.zero_cost_volume:
                    relative_poses *= 0
                   #------------------------------
                    if self.opt.post_process:
                        raise NotImplementedError
                    min_depth_bin = self.min_depth_tracker
                    max_depth_bin = self.max_depth_tracker

                    # -------------------------simin_model------------------------------------------
                    if self.opt.use_simmim:
                        _, _, masknet_featlst = self.models['simmim'
                        ](input_color, input_color.to(self.device), None)
                    # -------------------------simin_model------------------------------------------

                    output, lowest_cost, costvol = self.models["encoder"](input_color, lookup_frames,
                                                           relative_poses,
                                                           K,
                                                           invK,
                                                           masknet_featlst,
                                                           min_depth_bin, max_depth_bin)
                    output = self.models["depth"](output)
                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                gt_depths.append(input_depth)


        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)#(150, 128, 416)


        errors = []
        ratios = []
        dmax = 0
        for i in range(pred_disps.shape[0]):
            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = np.squeeze(pred_disps[i])
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp
            #==========================================================================================
            # depth_min = pred_disp.min()
            depth_max = pred_disp.max()
            dmax = depth_max
            #
            # max_val =  255
            # # pred_depth = 255 * (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min()+1e-8)
            # if depth_max - depth_min > np.finfo("float").eps:
            #     pred_depth = max_val * (pred_disp - depth_min) / (depth_max - depth_min)
            # else:
            #     pred_depth = np.zeros(pred_disp.shape, dtype=pred_disp.dtype)
            #==========================================================================================
            # if self.opt.eval_split == 'fsea':
            #     mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            # else:
            #     mask = gt_depth > 0
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(gt_depth, pred_depth))
        mean_errors = np.array(errors).mean(0)
        logging.info("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                            "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        logging.info(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        logging.info("\n  " + ("{:>8} | " * 3).format("usod10k-maxdepth","min_depth_bin",
                                                      "max_depth_bin"))
        logging.info(("&{: 8.3f}  " * 3).format(dmax,self.min_depth_tracker,self.max_depth_tracker) + "\\\\")

        logging.info("\n-> Done!")

        self.set_train()

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        mono_outputs = {}
        outputs = {}

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, None)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None)
        outputs.update(pose_pred)
        mono_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]#b 4 4
        relative_poses = torch.stack(relative_poses, 1)#b 1 4 4

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]#查找帧
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.opt.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will
                # skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        min_depth_bin = self.min_depth_tracker
        max_depth_bin = self.max_depth_tracker

        #-------------------------simin_model----------------------------------------------------------------------------
        if self.opt.use_simmim:
            loss_masknet,out_imgslst, masknet_featlst = self.models['simmim'](inputs['color_aug', 0, 0],inputs['color', 0, 0],inputs['mask'])
            # loss_masknet, out_imgslst, masknet_featlst = self.models['simmim'](inputs['color_aug', 0, 0],
            #                                                                    inputs['color_aug', 0, 0], inputs['mask'])
            
        # -------------------------simin_model----------------------------------------------------------------------------
        # single frame path
        if self.train_teacher_and_pose:
            feats = self.models["mono_encoder"](inputs["color_aug", 0, 0],masknet_featlst)
            mono_outputs.update(self.models['mono_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["mono_encoder"](inputs["color_aug", 0, 0],masknet_featlst)
                mono_outputs.update(self.models['mono_depth'](feats))

        self.generate_images_pred(inputs, mono_outputs)
        mono_losses = self.compute_losses(inputs, mono_outputs, is_multi=False)

        # update multi frame outputs dictionary with single frame outputs
        for key in list(mono_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'mono_' + key[0]
                _key = tuple(_key)
                outputs[_key] = mono_outputs[key]

        # multi frame path
        features, lowest_cost, confidence_mask = self.models["encoder"](inputs["color_aug", 0, 0],
                                                                        lookup_frames,
                                                                        relative_poses,
                                                                        inputs[('K', 2)],
                                                                        inputs[('inv_K', 2)],
                                                                        masknet_featlst,
                                                                        min_depth_bin=min_depth_bin,
                                                                        max_depth_bin=max_depth_bin,
                                                                        )
        outputs.update(self.models["depth"](features))
        outputs.update(out_imgslst)
        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.opt.height, self.opt.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.opt.height, self.opt.width],
                                                    mode="nearest")[:, 0]

        if not self.opt.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                           self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs, is_multi=True)
        losses = self.compute_losses(inputs, outputs, is_multi=True)

        # update losses with single frame losses
        if self.train_teacher_and_pose:
            for key, val in mono_losses.items():
                losses[key] += val

        # update adaptive depth bins
        if self.train_teacher_and_pose:
            self.update_adaptive_depth_bins(outputs)

        losses["mask_transform_loss"] = loss_masknet
        losses["loss"] = losses["loss"] + loss_masknet*0.3

        return outputs, losses

    def  generate_images_pred(self, inputs, outputs, is_multi=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]
                if is_multi:
                    # don't update posenet based on multi frame prediction
                    T = T.detach()

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    def update_adaptive_depth_bins(self, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""

        min_depth = outputs[('mono_depth', 0, 0)].detach().min(-1)[0].min(-1)[0]
        max_depth = outputs[('mono_depth', 0, 0)].detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        # increase range slightly
        min_depth = max(self.opt.min_depth, min_depth * 0.9)
        max_depth = max_depth * 1.1

        self.max_depth_tracker = self.max_depth_tracker * 0.99 + max_depth * 0.01
        self.min_depth_tracker = self.min_depth_tracker * 0.99 + min_depth * 0.01
    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # predict poses for reprojection loss
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.matching_ids}
            with torch.no_grad():
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            logging.info('----------- ONLY ACCEPT TWO FRAME INPUTS -----------')
            raise NotImplementedError

        return outputs
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask
    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, monocular network"""

        mono_output = outputs[('mono_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - mono_output) / mono_output) < 1.0
        mask *= ((mono_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]
    def compute_losses(self, inputs, outputs, is_multi=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss,
                                                             identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            if is_multi:
                reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
                if not self.opt.disable_motion_masking:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              outputs['consistency_mask'].unsqueeze(1))
                if not self.opt.no_matching_augmentation:
                    reprojection_loss_mask = (reprojection_loss_mask *
                                              (1 - outputs['augmentation_mask']))
                consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage multi frame prediction to be like singe frame where masking is happening
            if is_multi:
                multi_depth = outputs[("depth", 0, scale)]
                # no gradients for mono prediction!
                mono_depth = outputs[("mono_depth", 0, scale)].detach()
                consistency_loss = torch.abs(multi_depth - mono_depth) * consistency_mask
                consistency_loss = consistency_loss.mean()

                # save for logging to tensorboard
                consistency_target = (mono_depth.detach() * consistency_mask +
                                      multi_depth.detach() * (1 - consistency_mask))
                consistency_target = 1 / consistency_target
                outputs["consistency_target/{}".format(scale)] = consistency_target
                losses['consistency_loss/{}'.format(scale)] = consistency_loss
            else:
                consistency_loss = 0

            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        # depth_pred = torch.clamp(F.interpolate(
        #     depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = torch.clamp(F.interpolate(
                 depth_pred, [608, 968], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        logging.info(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                         sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(3, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)#color,color_aug
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[("disp", s)][j, 0])
            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('mono_disp', s)][j, 0])
            writer.add_image(
                "disp_mono/{}".format(j),
                disp, self.step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

                consistency_mask = \
                    outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)
                writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                writer.add_image(
                    "lowest_cost_masked/{}".format(j),
                    lowest_cost * consistency_mask, self.step)
                writer.add_image(
                    "consistency_mask/{}".format(j),
                    consistency_mask, self.step)


                consistency_target = colormap(outputs["consistency_target/0"][j].squeeze(0))
                writer.add_image(
                    "consistency_target/{}".format(j),
                    consistency_target, self.step)


                writer.add_image("Input/Augmented/{}".format(j),denormalize(outputs['x_aug'][j].data), self.step)
                writer.add_image("Input/Clear/{}".format(j), denormalize(outputs['x_clear'][j].data), self.step)
                writer.add_image("Output/Reconstructed/{}".format(j), denormalize(outputs['x_clear_rec'][j].data), self.step)
                writer.add_image("Mask/{}".format(j),  outputs['mask'][j].data, self.step)
                # 可视化 Loss Map
                loss_map = denormalize(outputs['x_clear_rec'] * outputs['mask'].detach())
                writer.add_image("Loss_Map/{}".format(j), loss_map[j], self.step, dataformats="CHW")

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if save_step:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch,self.step))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = self.min_depth_tracker
                to_save['max_depth_bin'] = self.max_depth_tracker
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
    def load_mono_model(self):
        model_list = ['pose_encoder', 'pose', 'mono_encoder', 'mono_depth']
        for n in model_list:
            print('loading {}'.format(n))
            path = os.path.join(self.opt.mono_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model(self):
        """Load model(s) from disk
        """
        #self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)

            if n == 'encoder':
                min_depth_bin = pretrained_dict.get('min_depth_bin')
                max_depth_bin = pretrained_dict.get('max_depth_bin')
                print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
                if min_depth_bin is not None:
                    # recompute bins
                    print('setting depth bins!')
                    self.models['encoder'].compute_depth_bins(min_depth_bin, max_depth_bin)

                    self.min_depth_tracker = min_depth_bin
                    self.max_depth_tracker = max_depth_bin

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                logging.info("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                logging.info("Can't load Adam - using random")
        else:
            logging.info("Cannot find Adam weights so Adam is randomly initialized")

        try:
            num_epoch = int(self.opt.load_weights_folder.split('_')[1])
            logging.info(f'========== Loading Contains Epoch Number ==========: {num_epoch}')
            for i in range(num_epoch):
                self.model_lr_scheduler.step()
        except:
            logging.info('========== Loading Doesnt Contain Epoch Number ==========')
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
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

def sum_params(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = p.cpu().data.numpy()
        s.append(np.sum(n))
    return sum(s)
def denormalize(x, mean=0.45, std=0.225):
    """
    将标准化的张量还原到 [0, 1] 范围，适用于 TensorBoard 可视化。
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x * std + mean  # 逆标准化
    return x
# def process_for_visualization(x):
#     """
#     将 Transformer 重建的图像处理为可视化格式
#     """
#     # if isinstance(x, torch.Tensor):
#     #     x = x.detach().cpu().numpy()
#     x = x.clamp(0, 1)  # 裁剪到 [0, 1]
#     if x.shape[-1] == 3:  # 如果通道在最后 (H, W, C)，需要调整格式
#         x = x.permute(2, 0, 1)
#     x = (x * 255).byte()  # 转换到 [0, 255]
#     return x