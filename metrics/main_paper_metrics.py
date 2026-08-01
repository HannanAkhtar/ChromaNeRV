import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim

from utils import rgb_to_ycbcr_bt709


def psnr_from_mse(mse):
    mse = float(mse)
    return float('inf') if mse == 0 else -10.0 * math.log10(mse)


def calculate_psnr(sum_squared_error, count):
    if count <= 0:
        raise ValueError('PSNR requires at least one sample')
    return psnr_from_mse(float(sum_squared_error) / int(count))


def weighted_yuv_psnr_dbavg(psnr_y, psnr_cb, psnr_cr):
    """Legacy diagnostic only; paper tables must use weighted_yuv_psnr_mse."""
    return (6.0 * psnr_y + psnr_cb + psnr_cr) / 8.0


def weighted_yuv_psnr_mse(mse_y, mse_cb, mse_cr):
    return psnr_from_mse((6.0 * mse_y + mse_cb + mse_cr) / 8.0)


def weighted_yuv_ssim(ssim_y, ssim_cb, ssim_cr):
    return (6.0 * ssim_y + ssim_cb + ssim_cr) / 8.0


def frame_psnr_values(error_tensor):
    per_frame_mse = error_tensor.square().flatten(1).mean(dim=1).detach().cpu().numpy()
    return [psnr_from_mse(float(mse)) for mse in per_frame_mse]


def temporal_rgb_error_diff_from_errors(errors):
    if len(errors) < 2:
        return float('nan')
    return float(np.mean([
        (current - previous).abs().mean().item()
        for previous, current in zip(errors[:-1], errors[1:])
    ]))


def safe_ms_ssim(output, target):
    try:
        return ms_ssim(output, target, data_range=1.0, size_average=True).item()
    except AssertionError:
        return ssim(output, target, data_range=1.0, size_average=True).item()


def safe_ssim(output, target):
    try:
        return ssim(output, target, data_range=1.0, size_average=True).item()
    except AssertionError:
        return float('nan')


def build_lpips_model(device):
    try:
        import lpips
        model = lpips.LPIPS(net='alex').to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f'LPIPS unavailable; writing NaN. Reason: {exc}')
        return None


def build_dists_model(device):
    try:
        from DISTS_pytorch import DISTS
        model = DISTS().to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f'DISTS unavailable; writing NaN. Reason: {exc}')
        return None


def evaluate_lpips(model, output, target):
    if model is None:
        return float('nan')
    return model(output * 2.0 - 1.0, target * 2.0 - 1.0).mean().item()


def evaluate_dists(model, output, target):
    if model is None:
        return float('nan')
    return model(output, target).mean().item()


def ffmpeg_libvmaf_info(ffmpeg='ffmpeg'):
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise RuntimeError(
            'VMAF requested, but FFmpeg was not found. Install an FFmpeg build '
            'that includes the libvmaf filter or run the launcher with --skip_vmaf.')
    version = subprocess.run(
        [executable, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    filters = subprocess.run(
        [executable, '-hide_banner', '-filters'],
        capture_output=True,
        text=True,
        check=True,
    )
    if 'libvmaf' not in filters.stdout and 'libvmaf' not in filters.stderr:
        raise RuntimeError(
            f'VMAF requested, but {version} does not expose the libvmaf filter. '
            'Install an FFmpeg build compiled with --enable-libvmaf or use --skip_vmaf.')
    return {'executable': executable, 'ffmpeg_version': version, 'filter': 'libvmaf'}


def compute_vmaf(
        reference_pattern,
        distorted_pattern,
        output_json,
        command_path,
        fps=30,
        model_path=None,
        ffmpeg='ffmpeg'):
    info = ffmpeg_libvmaf_info(ffmpeg)
    output_json = Path(output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    escaped_log_path = output_json.as_posix().replace(':', r'\:').replace("'", r"\'")
    filter_options = [
        f"log_fmt=json",
        f"log_path='{escaped_log_path}'",
    ]
    if model_path:
        escaped_model_path = (
            Path(model_path).resolve().as_posix().replace(':', r'\:').replace("'", r"\'")
        )
        filter_options.append(f"model=path='{escaped_model_path}'")
    filter_graph = (
        '[0:v]format=yuv420p[dist];[1:v]format=yuv420p[ref];'
        f'[dist][ref]libvmaf={":".join(filter_options)}'
    )
    command = [
        info['executable'], '-y',
        '-framerate', str(fps), '-i', str(distorted_pattern),
        '-framerate', str(fps), '-i', str(reference_pattern),
        '-lavfi', filter_graph, '-f', 'null', '-',
    ]
    Path(command_path).write_text(
        subprocess.list2cmdline(command) + '\n', encoding='utf-8')
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            'FFmpeg libvmaf evaluation failed. Command and stderr were saved at '
            f'{command_path}.\n{completed.stderr[-2000:]}')
    version_match = re.search(r'VMAF version\s+([^\s]+)', completed.stderr, re.IGNORECASE)
    with output_json.open('r', encoding='utf-8') as handle:
        report = json.load(handle)
    pooled = report.get('pooled_metrics', {}).get('vmaf', {})
    score = pooled.get('mean', pooled.get('harmonic_mean'))
    if score is None:
        raise RuntimeError(f'VMAF JSON did not contain a pooled score: {output_json}')
    info['vmaf_score'] = float(score)
    info['libvmaf_version'] = version_match.group(1) if version_match else 'not reported'
    return info


def compute_pooled_fid(reference_paths, reconstructed_paths, device='cpu', batch_size=16):
    if len(reference_paths) != len(reconstructed_paths):
        raise ValueError('FID reference and reconstruction sets must have equal sizes')
    if not reference_paths:
        raise ValueError('FID requires at least one image pair')
    try:
        from PIL import Image
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchvision.transforms.functional import to_tensor
    except Exception as exc:
        raise RuntimeError(
            'Pooled FID requires torchmetrics[image], torch-fidelity, Pillow, and torchvision.'
        ) from exc

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for start in range(0, len(reference_paths), batch_size):
        reference = torch.stack([
            to_tensor(Image.open(path).convert('RGB'))
            for path in reference_paths[start:start + batch_size]
        ]).to(device)
        reconstructed = torch.stack([
            to_tensor(Image.open(path).convert('RGB'))
            for path in reconstructed_paths[start:start + batch_size]
        ]).to(device)
        metric.update(reference, real=True)
        metric.update(reconstructed, real=False)
    return float(metric.compute().item())


def compute_pooled_kid(reference_paths, reconstructed_paths, device='cpu', batch_size=16):
    """Compute KID mean over paired image sets using torchmetrics."""
    if len(reference_paths) != len(reconstructed_paths):
        raise ValueError('KID reference and reconstruction sets must have equal sizes')
    if len(reference_paths) < 2:
        raise ValueError('KID requires at least two image pairs')
    try:
        from PIL import Image
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchvision.transforms.functional import to_tensor
    except Exception as exc:
        raise RuntimeError(
            'KID requires torchmetrics[image], torch-fidelity, Pillow, and torchvision.'
        ) from exc
    subset_size = min(50, len(reference_paths))
    metric = KernelInceptionDistance(
        feature=2048, subset_size=subset_size, normalize=True).to(device)
    for start in range(0, len(reference_paths), batch_size):
        reference = torch.stack([
            to_tensor(Image.open(path).convert('RGB'))
            for path in reference_paths[start:start + batch_size]
        ]).to(device)
        reconstructed = torch.stack([
            to_tensor(Image.open(path).convert('RGB'))
            for path in reconstructed_paths[start:start + batch_size]
        ]).to(device)
        metric.update(reference, real=True)
        metric.update(reconstructed, real=False)
    mean, standard_deviation = metric.compute()
    return {'kid_mean': float(mean.item()), 'kid_std': float(standard_deviation.item())}


class SequenceMetricAccumulator:
    """Accumulate the main paper sequence metrics without retaining full videos."""

    def __init__(self):
        self.rgb_sse = 0.0
        self.rgb_count = 0
        self.yuv_sse = [0.0, 0.0, 0.0]
        self.yuv_count = [0, 0, 0]
        self.rgb_ms_ssim = []
        self.component_ssim = [[], [], []]
        self.frame_rgb_psnr = []
        self.frame_y_psnr = []
        self.temporal_diffs = []
        self.previous_error = None

    def update(self, output, target):
        output = output.clamp(0, 1)
        target = target.clamp(0, 1)
        error = output - target
        self.rgb_sse += error.square().sum().item()
        self.rgb_count += error.numel()
        self.frame_rgb_psnr.extend(frame_psnr_values(error))

        output_yuv = rgb_to_ycbcr_bt709(output)
        target_yuv = rgb_to_ycbcr_bt709(target)
        yuv_error = output_yuv - target_yuv
        self.frame_y_psnr.extend(frame_psnr_values(yuv_error[:, :1]))
        for channel in range(3):
            channel_error = yuv_error[:, channel:channel + 1]
            self.yuv_sse[channel] += channel_error.square().sum().item()
            self.yuv_count[channel] += channel_error.numel()
            self.component_ssim[channel].append(
                safe_ssim(
                    output_yuv[:, channel:channel + 1],
                    target_yuv[:, channel:channel + 1],
                ))
        for frame in range(output.size(0)):
            self.rgb_ms_ssim.append(
                safe_ms_ssim(output[frame:frame + 1], target[frame:frame + 1]))
            current = error[frame].detach().cpu()
            if self.previous_error is not None:
                self.temporal_diffs.append(
                    (current - self.previous_error).abs().mean().item())
            self.previous_error = current

    def compute(self):
        mse_rgb = self.rgb_sse / self.rgb_count
        mse_yuv = [
            total / count for total, count in zip(self.yuv_sse, self.yuv_count)
        ]
        psnr_yuv = [psnr_from_mse(value) for value in mse_yuv]
        ssim_yuv = [float(np.nanmean(values)) for values in self.component_ssim]
        return {
            'rgb_mse': mse_rgb,
            'rgb_psnr': psnr_from_mse(mse_rgb),
            'rgb_ms_ssim': float(np.nanmean(self.rgb_ms_ssim)),
            'mse_y': mse_yuv[0],
            'mse_cb': mse_yuv[1],
            'mse_cr': mse_yuv[2],
            'psnr_y': psnr_yuv[0],
            'psnr_cb': psnr_yuv[1],
            'psnr_cr': psnr_yuv[2],
            'yuv_psnr_611_mse': weighted_yuv_psnr_mse(*mse_yuv),
            'yuv_psnr_611_dbavg': weighted_yuv_psnr_dbavg(*psnr_yuv),
            'ssim_y': ssim_yuv[0],
            'ssim_cb': ssim_yuv[1],
            'ssim_cr': ssim_yuv[2],
            'yuv_ssim_611': weighted_yuv_ssim(*ssim_yuv),
            'frame_psnr_mean': float(np.mean(self.frame_rgb_psnr)),
            'frame_psnr_std': float(np.std(self.frame_rgb_psnr)),
            'frame_y_psnr_mean': float(np.mean(self.frame_y_psnr)),
            'frame_y_psnr_std': float(np.std(self.frame_y_psnr)),
            'temporal_rgb_error_diff': (
                float(np.mean(self.temporal_diffs))
                if self.temporal_diffs else float('nan')
            ),
        }
