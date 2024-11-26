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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments.combined import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams

from scene.combined.gaussian_model_combined import GaussianModelCombined
from scene.combined import SceneCombined

from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal

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




