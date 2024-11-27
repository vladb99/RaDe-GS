#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import math

from scene.combined import GaussianModelCombined

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils.sh_utils import eval_sh


def render(
        viewpoint_camera,
        pc : GaussianModelCombined,
        pipe,
        bg_color : torch.Tensor,
        kernel_size,
        scaling_modifier = 1.0,
        require_coord : bool = True,
        require_depth : bool = True,
        override_color = None,
        cam_no=None,
        iter=None,
        num_down_emb_c=5,
        num_down_emb_f=5):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=torch.tensor(viewpoint_camera.image_height).cuda(),
        image_width=torch.tensor(viewpoint_camera.image_width).cuda(),
        tanfovx=torch.tensor(tanfovx).cuda(),
        tanfovy=torch.tensor(tanfovy).cuda(),
        kernel_size=kernel_size,
        bg=bg_color.cuda(),
        scale_modifier=torch.tensor(scaling_modifier).cuda(),
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=torch.tensor(pc.active_sh_degree).cuda(),
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        require_coord=require_coord,
        require_depth=require_depth,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    means3D_final, scales_final, rotations_final, opacity_final, shs_final, extras = pc._deformation(means3D, scales,
                                                                                                     rotations, opacity,
                                                                                                     time, cam_no, pc,
                                                                                                     None, shs,
                                                                                                     iter=iter,
                                                                                                     num_down_emb_c=num_down_emb_c,
                                                                                                     num_down_emb_f=num_down_emb_f)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "expected_coord": rendered_expected_coord,
            "median_coord": rendered_median_coord,
            "expected_depth": rendered_expected_depth,
            "median_depth": rendered_median_depth,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii,
            "normal": rendered_normal,
            # TODO: E-D3DGS also returns a depth. Is rendered_expected_depth or rendered_median_depth correct here? Or maybe none of them?
            "depth": rendered_expected_depth,
            "sh_coefs_final": shs_final,
            "extras": extras
    }

