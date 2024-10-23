import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss,  ssim
from gaussian_renderer import render
from scene import Scene
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import  Namespace
import gc
import json
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import path
from shutil import copyfile
from typing import Dict, Tuple

import torch
from tqdm import tqdm
import nvtx

# %%
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from compression.vq import CompressionSettings, compress_gaussians
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from finetune import finetune as finetune_no_mlp
from finetune_mlp import finetune
from utils.image_utils import psnr
from utils.loss_utils import ssim
import scene.diff_idx as diff_idx
import torchvision

def calc_importance(
    gaussians: GaussianModel, scene, pipeline_params
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling = gaussians.scaling_qa(
        gaussians.scaling_activation(gaussians._scaling.detach())
    )
    cov3d = gaussians.covariance_activation(
        scaling, 1.0, gaussians.get_rotation.detach(), True
    ).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(
        gaussians.scaling_factor_qa(gaussians._scaling_factor.detach())
    )

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0
    for camera in tqdm(scene.getTrainCameras(), desc="Calculating sensitivity"):
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = render(
            camera,
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d_scaled,
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]

    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels
    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach(), cov_grad.detach()

def initialize_gaussians(model_params, comp_params):
    gaussians = GaussianModel(model_params.sh_degree if model_params.sh_degree else 3, quantization=True)
    scene = Scene(model_params, gaussians, load_iteration="", shuffle=True)
    return gaussians, scene

def initial_compress(gaussians, scene, model_params, pipeline_params, optim_params, comp_params):
    print("Setting up compressed model...")
    color_importance, gaussian_sensitivity = calc_importance(gaussians, scene, pipeline_params)
    with torch.no_grad():
        color_importance_n = color_importance.amax(-1)
        gaussian_importance_n = gaussian_sensitivity.amax(-1)
        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=comp_params.color_importance_include,
            steps=int(comp_params.color_cluster_iterations),
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=comp_params.gaussian_importance_include,
            steps=int(comp_params.gaussian_cluster_iterations),
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )
        
        compress_gaussians(
            gaussians,
            color_importance_n,
            gaussian_importance_n,
            color_compression_settings if not comp_params.not_compress_color else None,
            gaussian_compression_settings
            if not comp_params.not_compress_gaussians
            else None,
            comp_params.color_compress_non_dir,
            prune_threshold=comp_params.prune_threshold,
        )

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)

    model_params.model_path = comp_params.output_vq

    return gaussians, scene

def initialize_diff_indexing(scene, gaussians):
    gaussian_indices = gaussians._gaussian_indices
    feature_indices = gaussians._feature_indices

    diff_idx.initial_train(gaussians._gaussian_indices_mlp, gaussian_indices, n_epochs=20)
    diff_idx.initial_train(gaussians._feature_indices_mlp, feature_indices, n_epochs=50)

    # test feature indices
    out = diff_idx.model_inference(gaussians._feature_indices_mlp, gaussians._feature_indices)

def render_and_eval(
    gaussians: GaussianModel,
    scene: Scene,
    model_params: ModelParams,
    pipeline_params: PipelineParams,
) -> Dict[str, float]:
    with torch.no_grad():
        ssims = []
        psnrs = []
        lpipss = []

        views = scene.getTestCameras()

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for i, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline_params, background)[
                "render"
            ].unsqueeze(0)
            gt = view.original_image[0:3, :, :].unsqueeze(0)

            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type="vgg"))
            gc.collect()
            torch.cuda.empty_cache()
            torchvision.utils.save_image(gt, f"renders/{scene.model_name}/gt_{i}.png") 
            torchvision.utils.save_image(rendering, f"renders/{scene.model_name}/render_{i}.png") 

        return {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }




if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--no_mlp", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)

    comp_params.finetune_iterations = 0
    if not model_params.sh_degree:
        model_params.sh_degree = 3
    
    base_path = model_params.model_path

    for i in range(10, 17):
        print(f'---------------------------{base_path} codebook size {2 ** i}-----------------------------')
        comp_params.color_codebook_size = 2 ** i
        comp_params.gaussian_codebook_size = 2 ** i
        model_params.model_path = f"{base_path}_codebook_size_{2 ** i}"
        comp_params.output_vq = model_params.model_path

        gaussians, scene = initialize_gaussians(model_params, comp_params)
        scene.model_name = model_params.model_path.replace('./output/', '')

        if not os.path.exists(f"renders/{scene.model_name}/"):
            os.makedirs(f"renders/{scene.model_name}/", exist_ok=True)
        if not os.path.exists(f"renders/{scene.model_name}/training"):
            os.makedirs(f"renders/{scene.model_name}/training", exist_ok=True)

        gaussians, scene = initial_compress(gaussians, scene, model_params, pipeline_params, optim_params, comp_params)

        if not args.no_mlp:
            initialize_diff_indexing(scene, gaussians)

        comp_params.finetune_iterations = 3_000
        scene.loaded_iter = 0

        if not args.no_mlp:
            print("Not using MLP")
            finetune_no_mlp(
                scene,
                model_params,
                optim_params,
                comp_params,
                pipeline_params,
                testing_iterations=[-1],
                debug_from = -1
            )
        else:
            finetune(
                scene,
                model_params,
                optim_params,
                comp_params,
                pipeline_params,
                testing_iterations=[-1],
                debug_from = -1
            )

        iteration = comp_params.finetune_iterations
        out_file = path.join(
            comp_params.output_vq,
            f"point_cloud/iteration_{iteration}/point_cloud.npz",
        )
        gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)
        file_size = os.path.getsize(out_file) / 1024**2
        print(f"saved vq finetuned model to {out_file}")

        # eval model
        print("evaluating...")
        metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
        metrics["size"] = file_size
        print(metrics)
        with open(f"{comp_params.output_vq}/results.json","w") as f:
            json.dump({f"ours_{iteration}":metrics},f,indent=4)
