import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from metrics.main_paper_metrics import (
    compute_pooled_fid,
    compute_vmaf,
    evaluate_dists,
    evaluate_lpips,
    ffmpeg_libvmaf_info,
    weighted_yuv_psnr_dbavg,
    weighted_yuv_psnr_mse,
    weighted_yuv_ssim,
)
from model_chroma_nerv import downsample_chroma, reconstruct_rgb_from_y_and_chroma
from nerv_generalization import (
    PAPER_CONFIGS,
    UVGSequenceDataset,
    build_job_config,
    discover_sequence_frames,
    natural_sort_key,
    run_is_complete,
    write_json,
)
from persistence import stable_config_hash
from scripts.aggregate_nerv_generalization import equal_sequence_average
from train_chroma_nerv import (
    build_model,
    estimate_model_macs_and_gflops,
    predict,
    validate_prediction_shapes,
)
from utils import PositionalEncoding, rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709


def paper_args(config_name):
    preset = PAPER_CONFIGS[config_name]
    width = preset['branch_width'] or 96
    return SimpleNamespace(
        experiment=preset['experiment'],
        color_space=preset['color_space'],
        stem_dim_num='512_1',
        fc_hw_dim='8_16_26',
        expansion=1.0,
        num_blocks=1,
        norm='none',
        act='swish',
        reduction=2,
        conv_type='conv',
        strides=[5, 3, 2, 2, 2],
        single_res=True,
        lower_width=96,
        sigmoid=False,
        chroma_scale=2,
        chroma_downsample='area',
        chroma_upsample='bilinear',
        chroma_upsampler='bilinear',
        learned_upsampler_width=16,
        learned_upsampler_depth=2,
        learned_upsampler_residual=False,
        y_branch_width=width,
        rgb_branch_width=width,
        upper_branch_width=96,
        chroma_branch_width=96,
        data_root='UVG_extracted',
        target_height=960,
        target_width=1920,
        fps_warmup=1,
        fps_repeats=1,
    )


class NeRVGeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = PositionalEncoding('1.25_40')
        cls.input = torch.empty(
            1, cls.encoding.embed_length, device='meta')
        cls.models = {}
        cls.predictions = {}
        cls.complexity = {}
        for config_name in PAPER_CONFIGS:
            args = paper_args(config_name)
            model = build_model(args, cls.encoding.embed_length).to('meta')
            cls.models[config_name] = model
            cls.predictions[config_name] = predict(model, cls.input, args)
            cls.complexity[config_name] = estimate_model_macs_and_gflops(
                model, cls.input, args)

    def test_bt709_round_trip_and_special_values(self):
        random_rgb = torch.rand(2, 3, 17, 19)
        restored = ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(random_rgb))
        self.assertLess((restored - random_rgb).abs().max().item(), 1e-5)
        gray = torch.full((1, 3, 2, 2), 0.5)
        gray_yuv = rgb_to_ycbcr_bt709(gray)
        self.assertTrue(torch.allclose(gray_yuv[:, 1:], torch.full_like(gray_yuv[:, 1:], 0.5)))
        black_white = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).repeat(1, 3, 1, 1)
        round_trip = ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(black_white))
        self.assertTrue(torch.allclose(round_trip, black_white, atol=1e-6))

    def test_chroma_downsample_and_reconstruction_shapes(self):
        rgb = torch.rand(1, 3, 32, 64)
        yuv = rgb_to_ycbcr_bt709(rgb)
        low = downsample_chroma(yuv[:, 1:], scale=2, mode='area')
        self.assertEqual(low.shape, (1, 2, 16, 32))
        reconstructed = reconstruct_rgb_from_y_and_chroma(
            yuv[:, :1], low, chroma_upsampler='bilinear', clamp=False)
        self.assertEqual(reconstructed.shape, rgb.shape)
        expected = F.interpolate(low, size=(32, 64), mode='bilinear', align_corners=False)
        actual_yuv = rgb_to_ycbcr_bt709(reconstructed)
        self.assertLess((actual_yuv[:, 1:] - expected).abs().max().item(), 2e-5)

    def test_all_six_paper_output_shapes(self):
        for name, prediction in self.predictions.items():
            self.assertEqual(prediction['rgb'].shape, (1, 3, 960, 1920), name)
            if name.startswith('chroma_'):
                self.assertEqual(prediction['cbcr_low'].shape, (1, 2, 480, 960), name)

    def test_architecture_progression(self):
        config = build_job_config('Beauty', 'full_rgb', 132, 300, 1)
        height, width = 8, 16
        progression = [(height, width)]
        for stride in config['architecture']['strides']:
            height, width = height * stride, width * stride
            progression.append((height, width))
        self.assertEqual(
            progression,
            [(8, 16), (40, 80), (120, 240), (240, 480), (480, 960), (960, 1920)],
        )

    def test_final_resolution_assertion(self):
        args = paper_args('full_rgb')
        bad = {
            'rgb': torch.empty(1, 3, 10, 10),
            'ycbcr': torch.empty(1, 3, 10, 10),
            'cbcr_low': None,
        }
        with self.assertRaisesRegex(AssertionError, 'expected 960x1920'):
            validate_prediction_shapes(bad, args)

    def test_matched_parameter_counts(self):
        for width in (8, 4):
            rgb = self.models[f'rgbsplit_w{width}']
            chroma = self.models[f'chroma_w{width}']
            counts = [sum(p.numel() for p in model.parameters()) for model in (rgb, chroma)]
            relative = abs(counts[0] - counts[1]) / max(counts)
            self.assertLess(relative, 0.005)

    def test_matched_gflops_and_width_ordering(self):
        for width in (8, 4):
            values = [
                self.complexity[f'rgbsplit_w{width}'][1],
                self.complexity[f'chroma_w{width}'][1],
            ]
            self.assertLess(abs(values[0] - values[1]) / max(values), 0.01)
        full = self.complexity['full_rgb'][1]
        self.assertLess(self.complexity['rgbsplit_w4'][1], self.complexity['rgbsplit_w8'][1])
        self.assertLess(self.complexity['rgbsplit_w8'][1], full)

    def test_weighted_yuv_metrics(self):
        mse = (0.01, 0.04, 0.09)
        expected = -10 * math.log10((6 * mse[0] + mse[1] + mse[2]) / 8)
        self.assertAlmostEqual(weighted_yuv_psnr_mse(*mse), expected)
        component_psnr = [-10 * math.log10(value) for value in mse]
        self.assertNotAlmostEqual(
            weighted_yuv_psnr_mse(*mse),
            weighted_yuv_psnr_dbavg(*component_psnr),
        )
        self.assertAlmostEqual(weighted_yuv_ssim(0.9, 0.8, 0.7), 0.8625)

    def test_natural_sort_and_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / 'Beauty'
            sequence.mkdir()
            for index in range(1, 134):
                Image.new('RGB', (2, 1)).save(sequence / f'frame{index}.png')
            names = [path.name for path in discover_sequence_frames(directory, 'Beauty')]
            self.assertEqual(len(names), 132)
            self.assertEqual(names[:3], ['frame1.png', 'frame2.png', 'frame3.png'])
            self.assertEqual(names[-1], 'frame132.png')
            self.assertLess(
                natural_sort_key('frame9.png'),
                natural_sort_key('frame10.png'),
            )
            dataset = UVGSequenceDataset(
                directory, 'Beauty', target_height=1, target_width=2)
            self.assertEqual(len(dataset), 132)

    def test_production_center_crop_preserves_exact_rows_and_width(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / 'Beauty'
            sequence.mkdir()
            row_values = (np.arange(1080, dtype=np.uint16) % 256).astype(np.uint8)
            image = np.broadcast_to(
                row_values[:, None, None], (1080, 1920, 3)).copy()
            Image.fromarray(image, mode='RGB').save(sequence / 'frame1.png')

            dataset = UVGSequenceDataset(
                directory,
                'Beauty',
                max_frames=1,
                target_height=960,
                target_width=1920,
            )
            tensor, _ = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 960, 1920))
            output_bytes = (tensor * 255).round().to(torch.uint8)
            self.assertEqual(output_bytes[0, 0, 0].item(), 60)
            self.assertEqual(output_bytes[0, -1, 0].item(), 1019 % 256)
            self.assertEqual(output_bytes.shape[-1], 1920)
            self.assertEqual(dataset.preprocessing_metadata['preprocessing_mode'], 'center_crop')
            self.assertEqual(dataset.preprocessing_metadata['crop_top'], 60)
            self.assertEqual(dataset.preprocessing_metadata['crop_left'], 0)
            self.assertEqual(dataset.preprocessing_metadata['crop_height'], 960)
            self.assertEqual(dataset.preprocessing_metadata['crop_width'], 1920)
            self.assertFalse(dataset.preprocessing_metadata['resize_applied'])

    def test_already_correct_production_frame_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / 'Beauty'
            sequence.mkdir()
            image = np.zeros((960, 1920, 3), dtype=np.uint8)
            image[..., 0] = np.arange(1920, dtype=np.uint16) % 256
            image[..., 1] = 71
            image[..., 2] = 203
            Image.fromarray(image, mode='RGB').save(sequence / 'frame1.png')
            dataset = UVGSequenceDataset(
                directory, 'Beauty', max_frames=1)
            tensor, _ = dataset[0]
            restored = (
                (tensor * 255).round().to(torch.uint8)
                .permute(1, 2, 0).numpy()
            )
            self.assertTrue(np.array_equal(restored, image))
            self.assertEqual(dataset.preprocessing_metadata['preprocessing_mode'], 'none')
            self.assertFalse(dataset.preprocessing_metadata['resize_applied'])

    def test_unsupported_resolution_requires_explicit_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / 'Beauty'
            sequence.mkdir()
            Image.new('RGB', (200, 100), (10, 20, 30)).save(
                sequence / 'frame1.png')
            with self.assertRaisesRegex(
                    ValueError,
                    'Detected source resolution 100x200.*No implicit resizing'):
                UVGSequenceDataset(
                    directory,
                    'Beauty',
                    max_frames=1,
                    target_height=16,
                    target_width=32,
                )
            dataset = UVGSequenceDataset(
                directory,
                'Beauty',
                max_frames=1,
                target_height=16,
                target_width=32,
                allow_resize=True,
            )
            tensor, _ = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 16, 32))
            self.assertEqual(dataset.preprocessing_metadata['preprocessing_mode'], 'resize')
            self.assertTrue(dataset.preprocessing_metadata['resize_applied'])

    def test_preprocessing_changes_scientific_config_hash(self):
        base = {
            'sequence': 'Beauty',
            'source_resolution': [1080, 1920],
            'target_resolution': [960, 1920],
            'preprocessing_mode': 'center_crop',
            'crop_top': 60,
            'crop_left': 0,
            'crop_height': 960,
            'crop_width': 1920,
            'resize_applied': False,
        }
        shifted = dict(base, crop_top=59)
        resized = dict(base, preprocessing_mode='resize', resize_applied=True)
        self.assertNotEqual(stable_config_hash(base), stable_config_hash(shifted))
        self.assertNotEqual(stable_config_hash(base), stable_config_hash(resized))

    def test_incomplete_run_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {'key': 'value'}
            write_json(run_dir / 'config.json', config)
            (run_dir / 'model_final.pth').touch()
            self.assertFalse(run_is_complete(run_dir, config))
            write_json(run_dir / 'eval_metrics.json', {'rgb_psnr': 1})
            self.assertTrue(run_is_complete(run_dir, config))
            self.assertFalse(run_is_complete(run_dir, {'key': 'different'}))

    def test_equal_sequence_weighting(self):
        rows = [
            {'rgb_psnr': 10.0, 'frame_count': 2},
            {'rgb_psnr': 30.0, 'frame_count': 132},
        ]
        self.assertEqual(equal_sequence_average(rows, 'rgb_psnr'), 20.0)

    def test_vmaf_missing_dependency_is_clear(self):
        with mock.patch('metrics.main_paper_metrics.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg was not found'):
                ffmpeg_libvmaf_info()

    @unittest.skipUnless(importlib.util.find_spec('lpips'), 'lpips is not installed')
    def test_lpips_identical_image(self):
        import lpips
        model = lpips.LPIPS(net='alex').eval()
        image = torch.rand(1, 3, 64, 64)
        self.assertLess(abs(evaluate_lpips(model, image, image)), 1e-6)

    @unittest.skipUnless(
        importlib.util.find_spec('DISTS_pytorch'),
        'DISTS-pytorch is not installed',
    )
    def test_dists_identical_image(self):
        from DISTS_pytorch import DISTS
        model = DISTS().eval()
        image = torch.rand(1, 3, 64, 64)
        self.assertLess(abs(evaluate_dists(model, image, image)), 1e-6)

    @unittest.skipUnless(
        importlib.util.find_spec('torchmetrics'),
        'torchmetrics is not installed',
    )
    def test_identical_pooled_fid(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = Path(directory) / f'{index}.png'
                Image.new('RGB', (64, 64), (index * 40, 20, 200)).save(path)
                paths.append(path)
            self.assertLess(abs(compute_pooled_fid(paths, paths)), 1e-3)

    def test_vmaf_identical_sequence_when_available(self):
        try:
            ffmpeg_libvmaf_info()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                Image.new('RGB', (64, 64), (40 + index, 80, 120)).save(
                    root / f'frame_{index:06d}.png')
            result = compute_vmaf(
                root / 'frame_%06d.png',
                root / 'frame_%06d.png',
                root / 'vmaf.json',
                root / 'command.txt',
            )
            self.assertGreater(result['vmaf_score'], 97.0)


if __name__ == '__main__':
    unittest.main()
