import torch
import torch.nn as nn
import torch.nn.functional as F

from model_nerv import MLP, NeRVBlock
from utils import rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709


def resize_chroma(chroma, size, mode):
    """Resize CbCr tensors while handling interpolate modes consistently."""
    kwargs = {'size': size, 'mode': mode}
    if mode in {'bilinear', 'bicubic'}:
        kwargs['align_corners'] = False
    return F.interpolate(chroma, **kwargs)


def downsample_chroma(chroma, scale=2, mode='area'):
    """Downsample full-resolution CbCr by an integer spatial scale."""
    height, width = chroma.shape[-2:]
    if scale < 1:
        raise ValueError(f'Chroma scale must be positive, got {scale}')
    if height % scale or width % scale:
        raise ValueError(
            f'Chroma scale {scale} requires divisible spatial dimensions, got {height}x{width}')
    return resize_chroma(chroma, (height // scale, width // scale), mode)


def downsample_chroma_420(chroma, mode='area'):
    """Backward-compatible 4:2:0 downsampling wrapper."""
    return downsample_chroma(chroma, scale=2, mode=mode)


def reconstruct_rgb_from_y_and_chroma(
        y, cbcr_low, chroma_upsampler='bilinear', learned_upsampler=None, clamp=True):
    """Upsample low-resolution CbCr and reconstruct an RGB frame."""
    if chroma_upsampler == 'learned':
        if learned_upsampler is None:
            raise ValueError('A learned chroma upsampler module is required for learned reconstruction')
        cbcr_full = learned_upsampler(cbcr_low, target_size=y.shape[-2:])
    else:
        cbcr_full = resize_chroma(cbcr_low, y.shape[-2:], chroma_upsampler)
    ycbcr = torch.cat((y, cbcr_full), dim=-3)
    rgb = ycbcr_to_rgb_bt709(ycbcr)
    return rgb.clamp(0, 1) if clamp else rgb


def reconstruct_rgb_from_420(y, cbcr_low, chroma_upsample='bilinear', clamp=True):
    """Backward-compatible 4:2:0 RGB reconstruction wrapper."""
    return reconstruct_rgb_from_y_and_chroma(
        y, cbcr_low, chroma_upsampler=chroma_upsample, clamp=clamp)


def apply_posthoc_420_to_rgb(
        rgb, chroma_downsample='area', chroma_upsample='bilinear', return_components=False):
    """Apply ordinary codec-style 4:2:0 chroma subsampling to RGB frames."""
    ycbcr = rgb_to_ycbcr_bt709(rgb)
    y = ycbcr[:, :1]
    cbcr_low = downsample_chroma_420(ycbcr[:, 1:], chroma_downsample)
    cbcr_full = resize_chroma(cbcr_low, y.shape[-2:], chroma_upsample)
    rgb_420 = ycbcr_to_rgb_bt709(torch.cat((y, cbcr_full), dim=1)).clamp(0, 1)
    if return_components:
        return rgb_420, y, cbcr_low, cbcr_full
    return rgb_420


class LearnedChromaUpsampler(nn.Module):
    """Small full-resolution CNN that refines bilinearly upsampled CbCr."""

    def __init__(self, width=16, depth=2, residual=False):
        super().__init__()
        if width < 1:
            raise ValueError('learned upsampler width must be positive')
        if depth < 1:
            raise ValueError('learned upsampler depth must be positive')
        layers = []
        in_channels = 2
        for _ in range(depth):
            layers.extend([
                nn.Conv2d(in_channels, width, 3, 1, 1),
                nn.SiLU(inplace=True),
            ])
            in_channels = width
        layers.append(nn.Conv2d(width, 2, 3, 1, 1))
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, cbcr_low, target_size):
        base = resize_chroma(cbcr_low, target_size, mode='bilinear')
        correction = self.net(base)
        if self.residual:
            return (base + 0.1 * torch.tanh(correction)).clamp(0, 1)
        return torch.sigmoid(correction)


class ChromaGenerator(nn.Module):
    """NeRV generator variants that emit full-resolution Y and low-resolution CbCr."""

    def __init__(self, experiment, **kwargs):
        super().__init__()
        supported_experiments = {
            'neural420_shared',
            'neural420_split',
            'neural420_shared_learned_up',
            'neural420_early_chroma',
            'neural420_asym_y',
        }
        if experiment not in supported_experiments:
            raise ValueError(f'Unsupported ChromaGenerator experiment: {experiment}')
        self.experiment = experiment
        self.sigmoid = kwargs['sigmoid']
        self.chroma_scale = kwargs.get('chroma_scale', 2)
        self.chroma_resize_mode = kwargs.get('chroma_downsample', 'area')
        self.learned_upsampler = None
        if kwargs.get('chroma_upsampler') == 'learned':
            self.learned_upsampler = LearnedChromaUpsampler(
                width=kwargs.get('learned_upsampler_width', 16),
                depth=kwargs.get('learned_upsampler_depth', 2),
                residual=kwargs.get('learned_upsampler_residual', False),
            )

        stem_dim, stem_num = [int(x) for x in kwargs['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kwargs['fc_hw_dim'].split('_')]
        mlp_dims = (
            [kwargs['embed_length']]
            + [stem_dim] * stem_num
            + [self.fc_h * self.fc_w * self.fc_dim]
        )
        self.stem = MLP(dim_list=mlp_dims, act=kwargs['act'])

        stride_list = kwargs['stride_list']
        if not stride_list:
            raise ValueError('stride_list must contain at least one upsampling stage')

        if experiment in {
                'neural420_shared', 'neural420_shared_learned_up', 'neural420_early_chroma'}:
            self.shared_layers, ngf = self._build_stages(
                self.fc_dim, stride_list, 0, **kwargs)
            self.y_head = nn.Conv2d(ngf, 1, 1, 1, bias=kwargs['bias'])
            cbcr_stride = 1 if experiment == 'neural420_early_chroma' else 2
            self.cbcr_head = nn.Conv2d(ngf, 2, 1, cbcr_stride, bias=kwargs['bias'])
        else:
            self.shared_layers, ngf = self._build_stages(
                self.fc_dim, stride_list[:-1], 0, **kwargs)
            self.cbcr_head = nn.Conv2d(ngf, 2, 1, 1, bias=kwargs['bias'])
            if experiment == 'neural420_asym_y':
                y_branch_width = kwargs.get('y_branch_width', ngf)
                self.y_adapter = (
                    nn.Identity()
                    if y_branch_width == ngf
                    else nn.Conv2d(ngf, y_branch_width, 1, 1, bias=kwargs['bias'])
                )
                self.y_layers, ngf = self._build_fixed_width_stage(
                    y_branch_width, stride_list[-1], **kwargs)
            else:
                self.y_layers, ngf = self._build_stages(
                    ngf, stride_list[-1:], len(stride_list) - 1, **kwargs)
            self.y_head = nn.Conv2d(ngf, 1, 1, 1, bias=kwargs['bias'])

    @staticmethod
    def _build_stages(ngf, strides, stage_offset, **kwargs):
        layers = nn.ModuleList()
        for relative_index, stride in enumerate(strides):
            stage_index = stage_offset + relative_index
            if stage_index == 0:
                new_ngf = int(ngf * kwargs['expansion'])
            else:
                divisor = 1 if stride == 1 else kwargs['reduction']
                new_ngf = max(ngf // divisor, kwargs['lower_width'])

            for block_index in range(kwargs['num_blocks']):
                layers.append(NeRVBlock(
                    ngf=ngf,
                    new_ngf=new_ngf,
                    stride=1 if block_index else stride,
                    bias=kwargs['bias'],
                    norm=kwargs['norm'],
                    act=kwargs['act'],
                    conv_type=kwargs['conv_type'],
                ))
                ngf = new_ngf
        return layers, ngf

    @staticmethod
    def _build_fixed_width_stage(ngf, stride, **kwargs):
        layers = nn.ModuleList()
        width = ngf
        for block_index in range(kwargs['num_blocks']):
            layers.append(NeRVBlock(
                ngf=ngf,
                new_ngf=width,
                stride=1 if block_index else stride,
                bias=kwargs['bias'],
                norm=kwargs['norm'],
                act=kwargs['act'],
                conv_type=kwargs['conv_type'],
            ))
            ngf = width
        return layers, ngf

    def _normalize(self, tensor):
        return torch.sigmoid(tensor) if self.sigmoid else (torch.tanh(tensor) + 1) * 0.5

    def forward(self, inputs):
        features = self.stem(inputs)
        features = features.view(features.size(0), self.fc_dim, self.fc_h, self.fc_w)
        for layer in self.shared_layers:
            features = layer(features)

        if self.experiment in {'neural420_split', 'neural420_asym_y'}:
            cbcr_low = self._normalize(self.cbcr_head(features))
            if self.experiment == 'neural420_asym_y':
                features = self.y_adapter(features)
            for layer in self.y_layers:
                features = layer(features)
        else:
            cbcr_low = self._normalize(self.cbcr_head(features))

        y = self._normalize(self.y_head(features))
        target_size = (y.shape[-2] // self.chroma_scale, y.shape[-1] // self.chroma_scale)
        if cbcr_low.shape[-2:] != target_size:
            cbcr_low = resize_chroma(cbcr_low, target_size, self.chroma_resize_mode)
        return {'y': y, 'cbcr_low': cbcr_low}
