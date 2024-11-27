from argparse import ArgumentParser, Namespace
import torch
import numpy as np
import random
import sys
import os
import uuid
import pyvista as pv
from dreifus.pyvista import add_camera_frustum
from dreifus.matrix import Pose, Intrinsics
from dreifus.camera import CameraCoordinateConvention, PoseType
from tqdm import tqdm
from random import randint

from train import L1_loss_appearance

from arguments.combined import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams

from scene.combined.gaussian_model_combined import GaussianModelCombined
from scene.combined import SceneCombined
from scene.combined.cameras import Camera

from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal, depth_double_to_normal, point_double_to_normal
from utils.loss_utils import l1_loss, ssim
from utils.combined.extra_utils import o3d_knn, weighted_l2_loss_v2

from gaussian_renderer.combined import render

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, visualize):
    first_iter = 0

    tb_writer = prepare_output_and_logger()

    gaussians = GaussianModelCombined(dataset.sh_degree, hyper)
    scene = SceneCombined(dataset, gaussians, shuffle=dataset.shuffle, duration=hyper.total_num_frames)

    if visualize:
        viewpoint_stack = scene.getTrainCameras().copy()

        images = dict()
        serials = []
        world_2_cam_poses = dict()  # serial => world_2_cam_pose

        # viewpoint_stack doesn't contain 16 cameras anymore as in the static case, but 16 * N, where N is the number of frames.
        # cam_names keeps track, from which direction we already got an image, in order to visualize
        cam_names = []
        # 15, because we use 1 camera as test camera
        NUM_DIFFERENT_CAMERAS = 15
        index = 0

        while len(cam_names) < NUM_DIFFERENT_CAMERAS:
            viewpoint_cam = viewpoint_stack[index]
            tmp_cam = viewpoint_stack[index]
            index += 1

            cam_name = viewpoint_cam.image_name.split("/")[0] # e.g. cam00/0000.png
            if viewpoint_cam.image_name.split("/")[0] not in cam_names and viewpoint_cam.original_image is not None:
                cam_names.append(cam_name)
            else:
                continue

            serial = viewpoint_cam.image_name
            serials.append(serial)

            gt_image = viewpoint_cam.original_image
            images[serial] = gt_image.cpu().detach().numpy().transpose(1, 2, 0)

            world_2_cam_pose = Pose(matrix_or_rotation=viewpoint_cam.R.T, translation=viewpoint_cam.T,
                                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV, pose_type=PoseType.WORLD_2_CAM)
            world_2_cam_poses[serial] = world_2_cam_pose

        fy = fov2focal(tmp_cam.FoVy, tmp_cam.image_height)
        fx = fov2focal(tmp_cam.FoVx, tmp_cam.image_width)
        intrinsics = Intrinsics(fx, fy, 0, 0)

        # Visualize camera poses and images
        p = pv.Plotter()
        #add_coordinate_axes(p, scale=0.1)
        for serial in serials:
            add_camera_frustum(p, world_2_cam_poses[serial], intrinsics, image=images[serial])
        p.show()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log, ema_psnr_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0, 0.0

    # TODO: E-D3DGS doesn't do .copy(). Maybe because of CUDA memory issue?
    train_cams = scene.getTrainCameras().copy()
    test_cams = scene.getTestCameras().copy()

    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=train_cams)

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map

    viewpoint_stack = train_cams

    # Used for embedding regularization
    prev_num_pts = 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # TODO: E-D3DGS has a more complex camera sampling, however they say it also works well for random camera sampling.
        # Pick a random Camera and random frame
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter

        cam_no = viewpoint_cam.cam_no
        render_pkg = render(
            viewpoint_cam,
            gaussians, pipe,
            background,
            kernel_size,
            require_coord = require_coord and reg_kick_on,
            require_depth = require_depth and reg_kick_on,
            cam_no=cam_no,
            iter=iteration,
            num_down_emb_c=hyper.min_embeddings,
            num_down_emb_f=hyper.min_embeddings
        )

        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"])
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        if reg_kick_on:
            lambda_depth_normal = opt.lambda_depth_normal
            if require_depth:
                rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth,
                                                               rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord,
                                                               rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1 - depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[
                1].mean()
        else:
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # TODO, E-D3DGS does this a little bit different
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (
                    1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))

        # TODO, E-D3DGS doesn't not opacity reset, but uses the mean opacity of all gaussians to regularize
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal

        # From E-D3DGS
        # embedding reg using knn (https://github.com/JonathonLuiten/Dynamic3DGaussians)
        if prev_num_pts != gaussians._xyz.shape[0]:
            neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._xyz.detach().cpu().numpy(), 20)
            neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
            neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            prev_num_pts = gaussians._xyz.shape[0]

        emb = gaussians._embedding[:, None, :].repeat(1, 20, 1)
        emb_knn = gaussians._embedding[neighbor_indices]
        loss += opt.reg_coef * weighted_l2_loss_v2(emb, emb_knn, neighbor_weight)

        # smoothness reg on temporal embeddings
        if opt.coef_tv_temporal_embedding > 0:
            weights = gaussians._deformation.weight
            N, C = weights.shape
            first_difference = weights[1:, :] - weights[N - 1, :]
            second_difference = first_difference[1:, :] - first_difference[N - 2, :]
            loss += opt.coef_tv_temporal_embedding * torch.square(second_difference).mean()

        loss.backward()

        iter_end.record()

def prepare_output_and_logger():
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description='Training script parameters')

    setup_seed(6666)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 5_000, 7_000, 14_000, 20_000, 30_000, 45_000, 60_000, 80_000, 100_000, 120_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configs", type=str, default = "")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args),
             hyper=hp.extract(args),
             opt=op.extract(args),
             pipe=pp.extract(args),
             testing_iterations=args.test_iterations,
             saving_iterations=args.save_iterations,
             checkpoint_iterations=args.checkpoint_iterations,
             checkpoint=args.start_checkpoint,
             debug_from=args.debug_from,
             visualize=True)

    # All done
    print("\nTraining complete.")




