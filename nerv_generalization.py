import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


UVG_SEQUENCES = (
    'Beauty',
    'Bosphorus',
    'HoneyBee',
    'Jockey',
    'ReadySetGo',
    'ShakeNDry',
    'YachtRide',
)

PAPER_CONFIGS = {
    'full_rgb': {
        'experiment': 'rgb444',
        'color_space': 'rgb',
        'branch_width': None,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.0,
    },
    'full_ycbcr444': {
        'experiment': 'ycbcr444',
        'color_space': 'ycbcr',
        'branch_width': None,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.0,
    },
    'rgbsplit_w8': {
        'experiment': 'rgb_asym',
        'color_space': 'rgb',
        'branch_width': 8,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.0,
    },
    'chroma_w8': {
        'experiment': 'neural420_asym_y',
        'color_space': 'ycbcr',
        'branch_width': 8,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.1,
    },
    'rgbsplit_w4': {
        'experiment': 'rgb_asym',
        'color_space': 'rgb',
        'branch_width': 4,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.0,
    },
    'chroma_w4': {
        'experiment': 'neural420_asym_y',
        'color_space': 'ycbcr',
        'branch_width': 4,
        'lambda_y': 1.0,
        'lambda_c': 1.0,
        'lambda_rgb': 0.1,
    },
}

DEFAULT_CONFIGS = tuple(PAPER_CONFIGS)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', str(value))
    ]


def parse_csv_names(value, allowed, label):
    names = tuple(item.strip() for item in value.split(',') if item.strip())
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise ValueError(f'Unknown {label}: {", ".join(unknown)}')
    if not names:
        raise ValueError(f'At least one {label} must be selected')
    return names


def discover_sequence_frames(data_root, sequence, max_frames=132):
    sequence_dir = Path(data_root) / sequence
    if not sequence_dir.is_dir():
        case_insensitive = next(
            (
                path for path in Path(data_root).iterdir()
                if path.is_dir() and path.name.lower() == sequence.lower()
            ),
            None,
        ) if Path(data_root).is_dir() else None
        sequence_dir = case_insensitive or sequence_dir
    if not sequence_dir.is_dir() and Path(data_root).is_dir():
        if any(
                path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                for path in Path(data_root).iterdir()):
            sequence_dir = Path(data_root)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f'UVG sequence directory not found: {sequence_dir}')

    candidates = [sequence_dir, sequence_dir / 'frames']
    frame_dir = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir()
            and any(path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir())
        ),
        None,
    )
    if frame_dir is None:
        raise FileNotFoundError(
            f'No PNG/JPG/JPEG frames found in {sequence_dir} or {sequence_dir / "frames"}')

    all_frames = [
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    png_frames = [path for path in all_frames if path.suffix.lower() == '.png']
    selected_type = png_frames if png_frames else all_frames
    selected_type.sort(key=lambda path: natural_sort_key(path.name))
    if max_frames is not None:
        if len(selected_type) < max_frames:
            raise ValueError(
                f'{sequence} contains {len(selected_type)} usable frames, but '
                f'--max_frames {max_frames} requires exactly that many.')
        selected_type = selected_type[:max_frames]
    if not selected_type:
        raise ValueError(f'No frames selected for {sequence}')
    return selected_type


def inspect_frame_resolution(frame_paths):
    expected = None
    for path in frame_paths:
        with Image.open(path) as image:
            current = (image.height, image.width)
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(
                f'Inconsistent frame resolution: {path} is {current[0]}x{current[1]}, '
                f'expected {expected[0]}x{expected[1]}')
    return expected


def resolve_spatial_preprocessing(
        source_resolution,
        target_resolution,
        allow_resize=False):
    source_resolution = tuple(source_resolution)
    target_resolution = tuple(target_resolution)
    source_height, source_width = source_resolution
    target_height, target_width = target_resolution
    metadata = {
        'source_resolution': list(source_resolution),
        'target_resolution': list(target_resolution),
        'preprocessing_mode': 'none',
        'spatial_transform': 'none',
        'crop_top': 0,
        'crop_left': 0,
        'crop_height': target_height,
        'crop_width': target_width,
        'crop_parameters': None,
        'resize_applied': False,
    }
    if source_resolution == target_resolution:
        return metadata
    if source_resolution == (1080, 1920) and target_resolution == (960, 1920):
        crop_top = (source_height - target_height) // 2
        crop_left = (source_width - target_width) // 2
        metadata.update({
            'preprocessing_mode': 'center_crop',
            'spatial_transform': 'center_crop',
            'crop_top': crop_top,
            'crop_left': crop_left,
            'crop_height': target_height,
            'crop_width': target_width,
            'crop_parameters': {
                'top': crop_top,
                'left': crop_left,
                'height': target_height,
                'width': target_width,
            },
        })
        return metadata
    if allow_resize:
        metadata.update({
            'preprocessing_mode': 'resize',
            'spatial_transform': 'resize',
            'crop_height': source_height,
            'crop_width': source_width,
            'resize_applied': True,
        })
        return metadata
    raise ValueError(
        f'Detected source resolution {source_height}x{source_width}; requested '
        f'target resolution {target_height}x{target_width}. Supported production '
        'source resolutions are 960x1920 (unchanged) and 1080x1920 '
        '(deterministic center crop to 960x1920). No implicit resizing was '
        'performed. Use --allow_resize only for an intentional compatibility '
        'or debug run.')


class UVGSequenceDataset(Dataset):
    def __init__(
            self,
            data_root,
            sequence,
            max_frames=132,
            frame_gap=1,
            target_height=960,
            target_width=1920,
            allow_resize=False):
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.frame_gap = frame_gap
        self.frame_paths = discover_sequence_frames(data_root, sequence, max_frames=max_frames)
        self.source_resolution = inspect_frame_resolution(self.frame_paths)
        self.target_resolution = (target_height, target_width)
        self.allow_resize = allow_resize
        self.preprocessing_metadata = resolve_spatial_preprocessing(
            self.source_resolution,
            self.target_resolution,
            allow_resize,
        )
        self.selected_frame_names = [
            str(path.relative_to(self.data_root)).replace('\\', '/')
            for path in self.frame_paths
        ]

    def __len__(self):
        return (len(self.frame_paths) + self.frame_gap - 1) // self.frame_gap

    def __getitem__(self, index):
        source_index = index * self.frame_gap
        path = self.frame_paths[source_index]
        with Image.open(path) as image:
            image = image.convert('RGB')
            mode = self.preprocessing_metadata['preprocessing_mode']
            if mode == 'center_crop':
                top = self.preprocessing_metadata['crop_top']
                left = self.preprocessing_metadata['crop_left']
                height = self.preprocessing_metadata['crop_height']
                width = self.preprocessing_metadata['crop_width']
                image = image.crop((left, top, left + width, top + height))
            elif mode == 'resize':
                image = image.resize(
                    (self.target_resolution[1], self.target_resolution[0]),
                    Image.Resampling.BICUBIC,
                )
            tensor = TF.to_tensor(image)
        if tuple(tensor.shape[-2:]) != self.target_resolution:
            raise AssertionError(
                f'Preprocessed frame has shape {tuple(tensor.shape)}, expected '
                f'[3, {self.target_resolution[0]}, {self.target_resolution[1]}]')
        norm_index = torch.tensor(
            source_index / len(self.frame_paths),
            dtype=torch.float32,
        )
        return tensor, norm_index


def set_reproducibility(seed, deterministic=False, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def git_commit(repo_root):
    try:
        result = subprocess.run(
            ['git', '-c', f'safe.directory={Path(repo_root).resolve().as_posix()}',
             'rev-parse', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return 'unavailable'


def environment_info(repo_root='.'):
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'torchvision_version': __import__('torchvision').__version__,
        'cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_name': gpu_name,
        'git_commit': git_commit(repo_root),
    }


def write_json(path, value):
    def json_safe(item):
        if isinstance(item, dict):
            return {key: json_safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(child) for child in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def read_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def configs_match(path, expected):
    try:
        return read_json(path) == expected
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run_is_complete(run_dir, expected_config, checkpoint_policy='final'):
    run_dir = Path(run_dir)
    checkpoint_name = 'model_final.pth' if checkpoint_policy == 'final' else 'model_best.pth'
    return (
        (run_dir / checkpoint_name).is_file()
        and (run_dir / 'eval_metrics.json').is_file()
        and configs_match(run_dir / 'config.json', expected_config)
    )


@dataclass(frozen=True)
class PaperArchitecture:
    embed: str = '1.25_40'
    stem_dim_num: str = '512_1'
    fc_hw_dim: str = '8_16_26'
    expansion: float = 1.0
    reduction: int = 2
    lower_width: int = 96
    num_blocks: int = 1
    norm: str = 'none'
    act: str = 'swish'
    conv_type: str = 'conv'
    strides: tuple = (5, 3, 2, 2, 2)
    target_height: int = 960
    target_width: int = 1920


def build_job_config(sequence, config_name, max_frames, epochs, seed, smoke_test=False):
    preset = PAPER_CONFIGS[config_name]
    architecture = PaperArchitecture()
    if smoke_test:
        architecture = PaperArchitecture(
            embed='1.25_4',
            stem_dim_num='32_1',
            fc_hw_dim='2_4_4',
            lower_width=4,
            strides=(2, 2, 2),
            target_height=16,
            target_width=32,
        )
    return {
        'sequence': sequence,
        'config_name': config_name,
        'experiment': preset['experiment'],
        'color_space': preset['color_space'],
        'branch_width': preset['branch_width'],
        'lambda_y': preset['lambda_y'],
        'lambda_c': preset['lambda_c'],
        'lambda_rgb': preset['lambda_rgb'],
        'max_frames': max_frames,
        'epochs': epochs,
        'seed': seed,
        'architecture': {
            'embed': architecture.embed,
            'stem_dim_num': architecture.stem_dim_num,
            'fc_hw_dim': architecture.fc_hw_dim,
            'expansion': architecture.expansion,
            'reduction': architecture.reduction,
            'lower_width': architecture.lower_width,
            'num_blocks': architecture.num_blocks,
            'norm': architecture.norm,
            'act': architecture.act,
            'conv_type': architecture.conv_type,
            'strides': list(architecture.strides),
            'target_height': architecture.target_height,
            'target_width': architecture.target_width,
        },
        'checkpoint_policy': 'final',
        'smoke_test': smoke_test,
        'training': {
            'optimizer': 'Adam',
            'learning_rate': 5e-4,
            'betas': [0.5, 0.999],
            'weight_decay': 0.0,
            'schedule': 'cosine',
            'warmup_ratio': 0.2,
            'warmup_epochs': int(0.2 * epochs),
            'batch_size': 1,
            'loss_type': 'L2',
            'frame_gap': 1,
            'test_gap': 1,
            'shuffle': True,
        },
    }
