from __future__ import print_function

import argparse
import csv
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim, ssim
from torchvision.utils import save_image

from model_chroma_nerv import (
    ChromaGenerator,
    apply_posthoc_420_to_rgb,
    downsample_chroma,
    reconstruct_rgb_from_y_and_chroma,
)
from model_nerv import CustomDataSet, Generator
from utils import (
    PositionalEncoding,
    adjust_lr,
    loss_fn,
    model_output_to_rgb,
    rgb_target_to_model_space,
    rgb_to_ycbcr_bt709,
    worker_init_fn,
)


EXPERIMENTS_NEURAL_420 = {
    'neural420_shared',
    'neural420_split',
    'neural420_shared_learned_up',
    'neural420_early_chroma',
    'neural420_asym_y',
}
EXPERIMENTS_420 = {'posthoc420'} | EXPERIMENTS_NEURAL_420
CSV_FIELDS = [
    'timestamp',
    'run_name',
    'ablation_group',
    'experiment',
    'dataset',
    'epochs',
    'checkpoint',
    'color_space',
    'params_M',
    'checkpoint_size_MB',
    'model_fps',
    'end_to_end_fps',
    'estimated_gflops',
    'rgb_psnr',
    'rgb_ms_ssim',
    'psnr_y',
    'psnr_cb',
    'psnr_cr',
    'output_sample_ratio',
    'lambda_y',
    'lambda_c',
    'lambda_rgb',
    'chroma_scale',
    'chroma_upsampler',
    'learned_upsampler_width',
    'learned_upsampler_depth',
    'learned_upsampler_residual',
    'y_branch_width',
    'chroma_branch_width',
    'visual_dir',
    'out_dir',
    'notes',
]


def parse_args():
    parser = argparse.ArgumentParser(description='ChromaNeRV experiment runner')

    # Dataset
    parser.add_argument('--vid', default=[None], type=int, nargs='+')
    parser.add_argument('--frame_gap', type=int, default=1)
    parser.add_argument('--test_gap', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='UVG')
    parser.add_argument('--data_dir', type=str, default=None)

    # Experiment
    parser.add_argument(
        '--experiment',
        required=True,
        choices=[
            'rgb444',
            'ycbcr444',
            'posthoc420',
            'neural420_shared',
            'neural420_split',
            'neural420_shared_learned_up',
            'neural420_early_chroma',
            'neural420_asym_y',
        ],
    )
    parser.add_argument(
        '--color_space',
        default=None,
        choices=['rgb', 'ycbcr'],
        help='Checkpoint output space. For posthoc420 this selects the source checkpoint type.',
    )
    parser.add_argument('--lambda_y', type=float, default=1.0)
    parser.add_argument('--lambda_c', type=float, default=1.0)
    parser.add_argument('--lambda_rgb', type=float, default=0.0)
    parser.add_argument('--chroma_downsample', default='area', choices=['area', 'bilinear', 'bicubic'])
    parser.add_argument('--chroma_upsample', default='bilinear', choices=['nearest', 'bilinear', 'bicubic'])
    parser.add_argument(
        '--chroma_upsampler',
        default=None,
        choices=['nearest', 'bilinear', 'bicubic', 'learned'],
        help='Overrides --chroma_upsample. Use learned for the CNN chroma refiner.',
    )
    parser.add_argument('--learned_upsampler_width', type=int, default=16)
    parser.add_argument('--learned_upsampler_depth', type=int, default=2)
    parser.add_argument('--learned_upsampler_residual', action='store_true')
    parser.add_argument('--chroma_scale', type=int, default=2, choices=[2, 4])
    parser.add_argument('--y_branch_width', type=int, default=96)
    parser.add_argument('--chroma_branch_width', type=int, default=96)
    parser.add_argument('--ablation_group', default='')

    # NeRV architecture
    parser.add_argument('--embed', type=str, default='1.25_80')
    parser.add_argument('--stem_dim_num', type=str, default='1024_1')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128')
    parser.add_argument('--expansion', type=float, default=8)
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2])
    parser.add_argument('--num-blocks', type=int, default=1)
    parser.add_argument('--norm', default='none', choices=['none', 'bn', 'in'])
    parser.add_argument(
        '--act',
        type=str,
        default='gelu',
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'],
    )
    parser.add_argument('--lower-width', type=int, default=32)
    parser.add_argument('--single_res', action='store_true')
    parser.add_argument('--conv_type', default='conv', choices=['conv', 'deconv', 'bilinear'])
    parser.add_argument('--sigmoid', action='store_true')

    # Training
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=150)
    parser.add_argument('--warmup', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_type', type=str, default='cosine', choices=['cosine', 'step', 'const'])
    parser.add_argument('--lr_steps', default=[], type=float, nargs='+')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='L2')
    parser.add_argument('--lw', type=float, default=1.0)
    parser.add_argument('--manualSeed', type=int, default=1)

    # Evaluation and output
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--fps_warmup', type=int, default=2)
    parser.add_argument('--fps_repeats', type=int, default=10)
    parser.add_argument('--visual_frames', type=int, default=1)
    parser.add_argument('--weight', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--outf', default='output/chroma_nerv')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--results_csv', default='results/chroma420_bunny_ablation_v2.csv')
    parser.add_argument('--notes', default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-p', '--print-freq', default=50, type=int)
    return parser.parse_args()


def resolve_args(args):
    expected_space = {'rgb444': 'rgb', 'ycbcr444': 'ycbcr'}.get(args.experiment)
    if expected_space is not None and args.color_space not in {None, expected_space}:
        raise ValueError(f'{args.experiment} requires --color_space {expected_space}')
    if args.experiment in EXPERIMENTS_NEURAL_420 and args.color_space not in {None, 'ycbcr'}:
        raise ValueError(f'{args.experiment} predicts YCbCr and requires --color_space ycbcr')

    if expected_space is not None:
        args.color_space = expected_space
    elif args.experiment in EXPERIMENTS_NEURAL_420:
        args.color_space = 'ycbcr'
    else:
        args.color_space = args.color_space or 'rgb'

    args.chroma_upsampler = args.chroma_upsampler or args.chroma_upsample
    if args.experiment == 'neural420_shared_learned_up':
        args.chroma_upsampler = 'learned'

    args.warmup = int(args.warmup * args.epochs)
    args.run_name = args.run_name or f"{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.out_dir = os.path.join(args.outf, args.dataset.lower(), args.run_name)
    args.data_dir = args.data_dir or os.path.join('data', args.dataset.lower())
    if args.debug:
        args.eval_freq = 1
    return args


def model_kwargs(args, embed_length):
    return {
        'embed_length': embed_length,
        'stem_dim_num': args.stem_dim_num,
        'fc_hw_dim': args.fc_hw_dim,
        'expansion': args.expansion,
        'num_blocks': args.num_blocks,
        'norm': args.norm,
        'act': args.act,
        'bias': True,
        'reduction': args.reduction,
        'conv_type': args.conv_type,
        'stride_list': args.strides,
        'sin_res': args.single_res,
        'lower_width': args.lower_width,
        'sigmoid': args.sigmoid,
        'chroma_scale': args.chroma_scale,
        'chroma_downsample': args.chroma_downsample,
        'chroma_upsampler': args.chroma_upsampler,
        'learned_upsampler_width': args.learned_upsampler_width,
        'learned_upsampler_depth': args.learned_upsampler_depth,
        'learned_upsampler_residual': args.learned_upsampler_residual,
        'y_branch_width': args.y_branch_width,
        'chroma_branch_width': args.chroma_branch_width,
    }


def build_model(args, embed_length):
    kwargs = model_kwargs(args, embed_length)
    if args.experiment in EXPERIMENTS_NEURAL_420:
        return ChromaGenerator(args.experiment, **kwargs)
    return Generator(**kwargs)


def unwrap_state_dict(checkpoint):
    state_dict = checkpoint.get('state_dict', checkpoint)
    return {
        key.replace('module.', '').replace('blocks.0.', ''): value
        for key, value in state_dict.items()
    }


def load_checkpoint(path, model, optimizer=None, resume=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(unwrap_state_dict(checkpoint))
    if resume and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def save_checkpoint(path, model, optimizer, epoch, best_rgb_psnr, args):
    torch.save(
        {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_rgb_psnr': best_rgb_psnr,
            'experiment': args.experiment,
            'color_space': args.color_space,
            'args': vars(args),
        },
        path,
    )


def predict(model, embed_input, args):
    if args.experiment in EXPERIMENTS_NEURAL_420:
        output = model(embed_input)
        rgb = reconstruct_rgb_from_y_and_chroma(
            output['y'],
            output['cbcr_low'],
            chroma_upsampler=args.chroma_upsampler,
            learned_upsampler=getattr(model, 'learned_upsampler', None),
        )
        return {
            'rgb': rgb,
            'ycbcr': rgb_to_ycbcr_bt709(rgb),
            'cbcr_low': output['cbcr_low'],
        }

    model_output = model(embed_input)[-1]
    rgb = model_output_to_rgb(model_output, args.color_space, clamp=True)
    if args.experiment == 'posthoc420':
        rgb, _, cbcr_low, _ = apply_posthoc_420_to_rgb(
            rgb,
            args.chroma_downsample,
            args.chroma_upsample,
            return_components=True,
        )
        return {'rgb': rgb, 'ycbcr': rgb_to_ycbcr_bt709(rgb), 'cbcr_low': cbcr_low}
    return {'rgb': rgb, 'ycbcr': rgb_to_ycbcr_bt709(rgb), 'cbcr_low': None}


def compute_train_loss(model, embed_input, rgb_target, args):
    if args.experiment in EXPERIMENTS_NEURAL_420:
        output = model(embed_input)
        ycbcr_target = rgb_to_ycbcr_bt709(rgb_target)
        y_target = ycbcr_target[:, :1]
        cbcr_target_low = downsample_chroma(
            ycbcr_target[:, 1:], scale=args.chroma_scale, mode=args.chroma_downsample)
        loss_y = F.mse_loss(output['y'], y_target)
        loss_c = F.mse_loss(output['cbcr_low'], cbcr_target_low)
        rgb_output = reconstruct_rgb_from_y_and_chroma(
            output['y'],
            output['cbcr_low'],
            chroma_upsampler=args.chroma_upsampler,
            learned_upsampler=getattr(model, 'learned_upsampler', None),
        )
        loss_rgb = F.mse_loss(rgb_output, rgb_target)
        total = args.lambda_y * loss_y + args.lambda_c * loss_c + args.lambda_rgb * loss_rgb
        return total, {'loss_y': loss_y.item(), 'loss_c': loss_c.item(), 'loss_rgb': loss_rgb.item()}

    outputs = model(embed_input)
    model_target = rgb_target_to_model_space(rgb_target, args.color_space)
    targets = [F.adaptive_avg_pool2d(model_target, output.shape[-2:]) for output in outputs]
    losses = [loss_fn(output, target, args) for output, target in zip(outputs, targets)]
    losses = [loss * (args.lw if index < len(losses) - 1 else 1) for index, loss in enumerate(losses)]
    total = sum(losses)
    return total, {'loss': total.item()}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_fps(model, embed_input, args, device):
    for _ in range(args.fps_warmup):
        model(embed_input)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.fps_repeats):
        model(embed_input)
    synchronize(device)
    model_seconds = time.perf_counter() - start

    for _ in range(args.fps_warmup):
        predict(model, embed_input, args)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.fps_repeats):
        predict(model, embed_input, args)
    synchronize(device)
    end_to_end_seconds = time.perf_counter() - start

    frame_count = embed_input.size(0) * args.fps_repeats
    return frame_count / model_seconds, frame_count / end_to_end_seconds


@torch.no_grad()
def estimate_model_gflops(model, embed_input, args):
    """Estimate neural FLOPs for one frame from executed Linear and Conv layers."""
    operation_count = [0]
    hooks = []

    def count_linear(module, inputs, output):
        operation_count[0] += output.numel() * module.in_features * 2

    def count_conv2d(module, inputs, output):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops *= module.in_channels // module.groups
        operation_count[0] += output.numel() * kernel_ops * 2

    def count_conv_transpose2d(module, inputs, output):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops *= module.in_channels // module.groups
        operation_count[0] += output.numel() * kernel_ops * 2

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))
        elif isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d))
        elif isinstance(module, torch.nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(count_conv_transpose2d))
    try:
        predict(model, embed_input[:1], args)
    finally:
        for hook in hooks:
            hook.remove()
    return operation_count[0] / 1e9


def representation_sample_ratio(args):
    if args.experiment not in EXPERIMENTS_420:
        return 1.0
    chroma_scale = 2 if args.experiment == 'posthoc420' else args.chroma_scale
    return (1 + 2 / chroma_scale ** 2) / 3


def calculate_psnr(sum_squared_error, count):
    mse = sum_squared_error / count
    return float('inf') if mse == 0 else -10 * np.log10(mse)


def safe_ms_ssim(rgb_output, rgb_target):
    try:
        return ms_ssim(rgb_output, rgb_target, data_range=1, size_average=True).item()
    except AssertionError:
        # Small debug frames cannot pass through all MS-SSIM downsamplings.
        return ssim(rgb_output, rgb_target, data_range=1, size_average=True).item()


def save_visuals(prediction, rgb_target, ycbcr_target, visual_dir, start_index, args):
    os.makedirs(visual_dir, exist_ok=True)
    batch_size = rgb_target.size(0)
    for batch_index in range(batch_size):
        frame_index = start_index + batch_index
        if frame_index >= args.visual_frames:
            break
        suffix = f'{frame_index:04d}.png'
        rgb_pred = prediction['rgb'][batch_index]
        ycbcr_pred = prediction['ycbcr'][batch_index]
        save_image(rgb_pred, os.path.join(visual_dir, f'pred_rgb_{suffix}'))
        save_image(rgb_target[batch_index], os.path.join(visual_dir, f'gt_rgb_{suffix}'))
        save_image((rgb_pred - rgb_target[batch_index]).abs(), os.path.join(visual_dir, f'error_rgb_{suffix}'))
        for channel_index, channel_name in enumerate(['y', 'cb', 'cr']):
            save_image(ycbcr_pred[channel_index:channel_index + 1], os.path.join(visual_dir, f'pred_{channel_name}_{suffix}'))
            save_image(ycbcr_target[batch_index, channel_index:channel_index + 1], os.path.join(visual_dir, f'gt_{channel_name}_{suffix}'))

        if prediction['cbcr_low'] is not None:
            cbcr_low = prediction['cbcr_low'][batch_index]
            save_image(cbcr_low[0:1], os.path.join(visual_dir, f'pred_cb_low_{suffix}'))
            save_image(cbcr_low[1:2], os.path.join(visual_dir, f'pred_cr_low_{suffix}'))
            save_image(ycbcr_pred[1:2], os.path.join(visual_dir, f'pred_cb_full_{suffix}'))
            save_image(ycbcr_pred[2:3], os.path.join(visual_dir, f'pred_cr_full_{suffix}'))


def append_csv(metrics, checkpoint_path, visual_dir, args):
    csv_dir = os.path.dirname(args.results_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    write_header = not os.path.isfile(args.results_csv)
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 ** 2) if os.path.isfile(checkpoint_path) else 0
    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'run_name': args.run_name,
        'ablation_group': args.ablation_group,
        'experiment': args.experiment,
        'dataset': args.dataset,
        'epochs': metrics['epoch'],
        'checkpoint': checkpoint_path,
        'color_space': args.color_space,
        'params_M': f"{metrics['params_M']:.6f}",
        'checkpoint_size_MB': f'{checkpoint_size:.6f}',
        'model_fps': f"{metrics['model_fps']:.4f}",
        'end_to_end_fps': f"{metrics['end_to_end_fps']:.4f}",
        'estimated_gflops': f"{metrics['estimated_gflops']:.4f}",
        'rgb_psnr': f"{metrics['rgb_psnr']:.4f}",
        'rgb_ms_ssim': f"{metrics['rgb_ms_ssim']:.6f}",
        'psnr_y': f"{metrics['psnr_y']:.4f}",
        'psnr_cb': f"{metrics['psnr_cb']:.4f}",
        'psnr_cr': f"{metrics['psnr_cr']:.4f}",
        'output_sample_ratio': f"{metrics['output_sample_ratio']:.4f}",
        'lambda_y': args.lambda_y,
        'lambda_c': args.lambda_c,
        'lambda_rgb': args.lambda_rgb,
        'chroma_scale': 2 if args.experiment == 'posthoc420' else args.chroma_scale,
        'chroma_upsampler': args.chroma_upsampler,
        'learned_upsampler_width': args.learned_upsampler_width,
        'learned_upsampler_depth': args.learned_upsampler_depth,
        'learned_upsampler_residual': args.learned_upsampler_residual,
        'y_branch_width': args.y_branch_width,
        'chroma_branch_width': args.chroma_branch_width,
        'visual_dir': visual_dir,
        'out_dir': args.out_dir,
        'notes': args.notes,
    }
    with open(args.results_csv, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_metrics(metrics, checkpoint_path):
    return (
        f"{datetime.now().isoformat(timespec='seconds')}\n"
        f'Checkpoint: {checkpoint_path}\n'
        f"Experiment: {metrics['experiment']}\n"
        f"RGB PSNR: {metrics['rgb_psnr']:.4f}\n"
        f"RGB MS-SSIM: {metrics['rgb_ms_ssim']:.6f}\n"
        f"PSNR-Y: {metrics['psnr_y']:.4f}\n"
        f"PSNR-Cb: {metrics['psnr_cb']:.4f}\n"
        f"PSNR-Cr: {metrics['psnr_cr']:.4f}\n"
        f"Params: {metrics['params_M']:.6f}M\n"
        f"Estimated GFLOPs: {metrics['estimated_gflops']:.4f}\n"
        f"Model FPS: {metrics['model_fps']:.4f}\n"
        f"End-to-end FPS: {metrics['end_to_end_fps']:.4f}\n"
        f"Output sample ratio: {metrics['output_sample_ratio']:.4f}\n"
    )


@torch.no_grad()
def evaluate(model, dataloader, positional_encoding, device, checkpoint_path, epoch, args):
    model.eval()
    sums = {'rgb': 0.0, 'y': 0.0, 'cb': 0.0, 'cr': 0.0}
    counts = {'rgb': 0, 'y': 0, 'cb': 0, 'cr': 0}
    msssim_total = 0.0
    frame_count = 0
    benchmark_input = None
    visual_dir = os.path.join(args.out_dir, 'visuals')

    for step, (rgb_target, norm_index) in enumerate(dataloader):
        if args.debug and step > 1:
            break
        rgb_target = rgb_target.to(device, non_blocking=True)
        embed_input = positional_encoding(norm_index).to(device, non_blocking=True)
        prediction = predict(model, embed_input, args)
        ycbcr_target = rgb_to_ycbcr_bt709(rgb_target)

        rgb_error = prediction['rgb'] - rgb_target
        ycbcr_error = prediction['ycbcr'] - ycbcr_target
        sums['rgb'] += rgb_error.square().sum().item()
        counts['rgb'] += rgb_error.numel()
        for channel_index, channel_name in enumerate(['y', 'cb', 'cr']):
            channel_error = ycbcr_error[:, channel_index:channel_index + 1]
            sums[channel_name] += channel_error.square().sum().item()
            counts[channel_name] += channel_error.numel()
        msssim_total += safe_ms_ssim(prediction['rgb'], rgb_target) * rgb_target.size(0)
        save_visuals(prediction, rgb_target, ycbcr_target, visual_dir, frame_count, args)
        frame_count += rgb_target.size(0)
        benchmark_input = embed_input if benchmark_input is None else benchmark_input

    if frame_count == 0:
        raise RuntimeError('Evaluation dataset is empty')
    model_fps, end_to_end_fps = benchmark_fps(model, benchmark_input, args, device)
    estimated_gflops = estimate_model_gflops(model, benchmark_input, args)
    metrics = {
        'epoch': epoch,
        'experiment': args.experiment,
        'params_M': sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        'rgb_psnr': calculate_psnr(sums['rgb'], counts['rgb']),
        'rgb_ms_ssim': msssim_total / frame_count,
        'psnr_y': calculate_psnr(sums['y'], counts['y']),
        'psnr_cb': calculate_psnr(sums['cb'], counts['cb']),
        'psnr_cr': calculate_psnr(sums['cr'], counts['cr']),
        'model_fps': model_fps,
        'end_to_end_fps': end_to_end_fps,
        'estimated_gflops': estimated_gflops,
        'output_sample_ratio': representation_sample_ratio(args),
    }
    text = format_metrics(metrics, checkpoint_path)
    print(text)
    with open(os.path.join(args.out_dir, 'eval.txt'), 'a') as eval_file:
        eval_file.write(text + '\n')
    append_csv(metrics, checkpoint_path, visual_dir, args)
    return metrics


def build_dataloaders(args):
    transform = transforms.ToTensor()
    train_dataset = CustomDataSet(args.data_dir, transform, vid_list=args.vid, frame_gap=args.frame_gap)
    val_dataset = CustomDataSet(args.data_dir, transform, vid_list=args.vid, frame_gap=args.test_gap)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


def main():
    args = resolve_args(parse_args())
    if args.overwrite and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    cudnn.benchmark = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positional_encoding = PositionalEncoding(args.embed)
    model = build_model(args, positional_encoding.embed_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    latest_path = os.path.join(args.out_dir, 'model_latest.pth')

    checkpoint_path = args.weight
    if args.resume and checkpoint_path is None and os.path.isfile(latest_path):
        checkpoint_path = latest_path
    start_epoch = 0
    best_rgb_psnr = float('-inf')
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, resume=args.resume)
        if args.resume:
            start_epoch = checkpoint.get('epoch', 0)
            best_rgb_psnr = checkpoint.get('best_rgb_psnr', best_rgb_psnr)
        print(f'Loaded checkpoint: {checkpoint_path}')

    print(args)
    print(model)
    print(f'Device: {device}')
    print(f'Model params: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.6f}M')
    train_loader, val_loader = build_dataloaders(args)

    if args.eval_only:
        if checkpoint_path is None:
            raise ValueError('--eval_only requires --weight or a resumable checkpoint')
        evaluate(
            model,
            val_loader,
            positional_encoding,
            device,
            checkpoint_path,
            start_epoch,
            args,
        )
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, (rgb_target, norm_index) in enumerate(train_loader):
            if args.debug and step > 1:
                break
            rgb_target = rgb_target.to(device, non_blocking=True)
            embed_input = positional_encoding(norm_index).to(device, non_blocking=True)
            learning_rate = adjust_lr(optimizer, epoch, step, len(train_loader), args)
            optimizer.zero_grad()
            loss, components = compute_train_loss(model, embed_input, rgb_target, args)
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0 or step == len(train_loader) - 1:
                component_text = ', '.join(f'{key}={value:.6f}' for key, value in components.items())
                print(
                    f'Epoch [{epoch + 1}/{args.epochs}] step [{step + 1}/{len(train_loader)}] '
                    f'lr={learning_rate:.2e}, {component_text}',
                    flush=True,
                )

        save_checkpoint(latest_path, model, optimizer, epoch + 1, best_rgb_psnr, args)
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            metrics = evaluate(
                model,
                val_loader,
                positional_encoding,
                device,
                latest_path,
                epoch + 1,
                args,
            )
            if metrics['rgb_psnr'] > best_rgb_psnr:
                best_rgb_psnr = metrics['rgb_psnr']
                save_checkpoint(
                    os.path.join(args.out_dir, 'model_val_best.pth'),
                    model,
                    optimizer,
                    epoch + 1,
                    best_rgb_psnr,
                    args,
                )
                save_checkpoint(latest_path, model, optimizer, epoch + 1, best_rgb_psnr, args)


if __name__ == '__main__':
    main()
