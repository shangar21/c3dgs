import os
import gc
import torch
from random import randint
from utils.loss_utils import l1_loss,  ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import  Namespace
import json
import torchvision
from arguments import ModelParams, PipelineParams

def accumulate_gradients(grad_storage, model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in grad_storage:
                grad_storage[name] = {'gradients': []}
            grad_storage[name]['gradients'].append(param.grad.abs().clone().detach())

def calculate_statistics(grad_storage):
    gradients_mean = {}
    gradients_median = {}

    for name, data in grad_storage.items():
        stacked_grads = torch.stack(data['gradients'])
        gradients_mean[name] = stacked_grads.mean(dim=0).mean().item()
        gradients_median[name] = stacked_grads.median(dim=0).values.median().item()

    return gradients_mean, gradients_median

def finetune(scene: Scene, dataset, opt, comp, pipe, testing_iterations, debug_from, use_mlp_every=100):
    prepare_output_and_logger(comp.output_vq, dataset)
    
    first_iter = scene.loaded_iter
    max_iter = first_iter + comp.finetune_iterations

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    scene.gaussians.training_setup(opt)
    scene.gaussians.update_learning_rate(first_iter)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, max_iter), desc="Training progress")
    first_iter += 1
    psnr_track = []
    loss_track = []
    ssim_track = []
    grad_storage = {}
    log_interval = 200
    gradients_log = []
    count = 0
    use_mlp = False
    
    for iteration in range(first_iter, max_iter + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if count == use_mlp_every or use_mlp:
            count -= 1
            use_mlp = count > 0
            render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background, use_mlp=use_mlp)
        else:
            count += 1
            render_pkg = render(viewpoint_cam, scene.gaussians, pipe, background)

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        if use_mlp:
            # Disable gradients for all parameters
            for name, param in scene.gaussians.named_parameters():
                param.requires_grad = False
            
            # Enable gradients only for specific submodule (e.g., DifferentiableIndexing MLP)
            for name, module in scene.gaussians.named_modules():
                if 'mlp' in name:
                    for pname, param in module.named_parameters():
                        param.requires_grad = True
        else:
            for name, param in scene.gaussians.named_parameters():
                if 'mlp' not in name:
                    param.requires_grad = True    
                else:
                    param.requires_grad = False

        loss.backward()
        
        accumulate_gradients(grad_storage, scene.gaussians)

        if iteration % log_interval == 0:
            gradients_mean, gradients_median = calculate_statistics(grad_storage)
            gradients_log.append({
                'iteration': iteration,
                'gradients_mean': gradients_mean,
                'gradients_median': gradients_median
            })
            grad_storage.clear()
            json.dump(gradients_log, open(f"./output/{scene.model_name}/gradient_log.json", 'w+'))

        tmp = psnr(image, gt_image).mean().item()



        if iteration == first_iter or iteration % 5 == 0 or True:
            eval_results = {}
            
            test_psnrs = []
            test_losses = []
            test_ssims = []
            for tc in scene.getTestCameras():
                t_img = render(tc, scene.gaussians, pipe, background, use_mlp=False)["render"]
                test_psnrs.append(psnr(t_img, tc.original_image).mean().item())
                test_ssims.append(ssim(t_img, tc.original_image).mean().item())
                ll1_test = l1_loss(t_img, tc.original_image)
                loss_test = (1.0 - opt.lambda_dssim) * ll1_test + opt.lambda_dssim * (1.0 - ssim(t_img, tc.original_image))
                test_losses.append(loss_test.item())
            eval_results['PSNR'] = sum(test_psnrs) / len(test_psnrs)
            eval_results['LOSS'] = sum(test_losses) / len(test_losses)
            eval_results['SSIM'] = sum(test_ssims) / len(test_ssims)

            psnr_track.append(eval_results['PSNR'])
            json.dump(psnr_track, open(f"./output/{scene.model_name}/diff_idx_psnr.json", 'w+'))
            loss_track.append(eval_results["LOSS"])
            json.dump(loss_track, open(f"./output/{scene.model_name}/diff_idx_loss.json", 'w+'))
            ssim_track.append(eval_results['SSIM'])
            json.dump(loss_track, open(f"./output/{scene.model_name}/diff_idx_ssim.json", 'w+'))

        iter_end.record()
        scene.gaussians.update_learning_rate(iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", 'PSNR': tmp})
                progress_bar.update(10)
            if iteration == max_iter:
                progress_bar.close()

            # Optimizer step
            if iteration < max_iter:
                scene.gaussians.optimizer.step()
                scene.gaussians.optimizer.zero_grad()


def prepare_output_and_logger(output_folder, args):
    if not output_folder:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        output_folder = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(output_folder))
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
