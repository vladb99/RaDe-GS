import os
import sys
import numpy as np
from typing import NamedTuple
from PIL import Image
from plyfile import PlyData

from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from scene.combined.gaussian_model_combined import BasicPointCloud

from utils.graphics_utils import focal2fov, getWorld2View2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array
    hpdirecitons: np.array
    cxr: float
    cyr: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readColmapSceneInfoNersemble(path, images, eval, duration, testonly=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100

    cam_infos_unsorted = readColmapCamerasDynerf(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                 images_folder=path, near=near, far=far, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    video_cam_infos = buildTrajectory(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, near=near, far=far, duration=duration)
    train_cam_infos = [_ for _ in cam_infos if "cam00" not in _.image_name]
    test_cam_infos = [_ for _ in cam_infos if "cam00" in _.image_name]

    uniquecheck = []
    for cam_info in test_cam_infos:
        if cam_info.image_name[:5] not in uniquecheck:
            uniquecheck.append(cam_info.image_name[:5])
    assert len(uniquecheck) == 1

    sanitycheck = []
    for cam_info in train_cam_infos:
        if cam_info.image_name[:5] not in sanitycheck:
            sanitycheck.append(cam_info.image_name[:5])
    for testname in uniquecheck:
        assert testname not in sanitycheck

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D_downsample.ply")

    if not testonly:
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            print("error:", e)
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapCamerasDynerf(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=300):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime+int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            if j == startime:
                image = Image.open(image_path)
                image = image.resize((int(width), int(height)), Image.LANCZOS)
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)
            else:
                image = None
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

# similar to here https://github.com/tobias-kirschstein/nersemble/blob/master/scripts/render/render_nersemble.py#L62
def buildTrajectory(cam_extrinsics, cam_intrinsics, near, far, duration):
    c2ws_all = {}
    c2w_dreifus = {}
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(extr.qvec)
        w2c[:3, 3] = np.array(extr.tvec)
        c2w = np.linalg.inv(w2c)
        c2ws_all[key] = c2w[:3, :]

        c2w_dreifus[key] = Pose(c2w,
                                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                                pose_type=PoseType.CAM_2_WORLD)
    c2ws_all = np.stack([value for _, value in sorted(c2ws_all.items())])

    if intr.model == "SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model == "PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    # TODO compute center point only using training cameras
    translations = np.array([c2w.get_translation() for _, c2w in c2w_dreifus.items()])
    central_point_world = np.mean(translations, axis=0)
    move = central_point_world

    cameras_z_directions_world = [c2w.get_rotation_matrix() @ np.array([0, 0, 1]) for _, c2w in c2w_dreifus.items()]
    mean_z_direction_world = np.mean(cameras_z_directions_world, axis=0)
    look_at = central_point_world + 1.1 * mean_z_direction_world

    # TODO check FPS of dataset
    n_timesteps = int(duration)
    cam_2_world_poses = circle_around_axis(n_timesteps,
                                           axis=Vec3(mean_z_direction_world[0], mean_z_direction_world[1], mean_z_direction_world[2]),
                                           up=Vec3(0, -1, 0),
                                           move=Vec3(move[0], move[1], move[2]),
                                           look_at=look_at,
                                           distance=0.5)
    cam_2_world_poses = [pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_CV, inplace=False) for
                         pose in cam_2_world_poses]
    cam_2_world_poses = np.stack(cam_2_world_poses)
    #cam_2_world_poses[:, :3, 3] *= 1

    height = intr.height
    width = intr.width
    cam_infos = []

    trajectory_dreifus = []

    for i, c2w in enumerate(cam_2_world_poses):
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image = None
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=None, image_name=None,
                              width=width, height=height, near=near, far=far, timestamp=i / (len(cam_2_world_poses) - 1),
                              pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
        cam_infos.append(cam_info)
        trajectory_dreifus.append(Pose(w2c,
                                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                                pose_type=PoseType.WORLD_2_CAM))
    sys.stdout.write('\n')

    """ Dreifus visualization
    intrinsics_dreifus = Intrinsics(intr.params[0], intr.params[1], 0, 0)
    # Visualize camera poses and images
    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    for _, pose in c2w_dreifus.items():
        add_camera_frustum(p, pose, intrinsics_dreifus, image=None)
    for pose in trajectory_dreifus:
        add_camera_frustum(p, pose, intrinsics_dreifus, image=None)
    p.show()
    """

    return cam_infos

sceneLoadTypeCallbacks = {
    "Nersemble": readColmapSceneInfoNersemble
}