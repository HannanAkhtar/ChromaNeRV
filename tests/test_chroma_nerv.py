import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model_chroma_nerv import (
    ChromaGenerator,
    LearnedChromaUpsampler,
    RGBEarlySplitUpperGenerator,
    RGBAsymGenerator,
    YUVEarlySplitUpperGenerator,
    apply_posthoc_420_to_rgb,
    downsample_chroma,
    downsample_chroma_420,
    reconstruct_rgb_from_y_and_chroma,
    reconstruct_rgb_from_420,
)
from train_chroma_nerv import build_model, estimate_model_gflops, representation_sample_ratio
from train_chroma_nerv import temporal_rgb_error_diff_from_errors
from train_chroma_nerv import weighted_yuv_psnr_dbavg, weighted_yuv_psnr_mse


def generator_kwargs():
    return {
        'embed_length': 4,
        'stem_dim_num': '8_1',
        'fc_hw_dim': '2_3_4',
        'expansion': 1,
        'num_blocks': 1,
        'norm': 'none',
        'act': 'gelu',
        'bias': True,
        'reduction': 2,
        'conv_type': 'conv',
        'stride_list': [2, 2],
        'sin_res': True,
        'lower_width': 2,
        'sigmoid': True,
    }


def predict_args(experiment):
    return SimpleNamespace(
        experiment=experiment,
        chroma_upsampler='bilinear',
        chroma_downsample='area',
        chroma_upsample='bilinear',
    )


def trainer_args(experiment):
    kwargs = generator_kwargs()
    return SimpleNamespace(
        experiment=experiment,
        stem_dim_num=kwargs['stem_dim_num'],
        fc_hw_dim=kwargs['fc_hw_dim'],
        expansion=kwargs['expansion'],
        num_blocks=kwargs['num_blocks'],
        norm=kwargs['norm'],
        act=kwargs['act'],
        reduction=kwargs['reduction'],
        conv_type=kwargs['conv_type'],
        strides=kwargs['stride_list'],
        single_res=kwargs['sin_res'],
        lower_width=kwargs['lower_width'],
        sigmoid=kwargs['sigmoid'],
        chroma_scale=2,
        chroma_downsample='area',
        chroma_upsampler='bilinear',
        learned_upsampler_width=16,
        learned_upsampler_depth=2,
        learned_upsampler_residual=False,
        y_branch_width=2,
        rgb_branch_width=2,
        upper_branch_width=2,
        chroma_branch_width=2,
    )


class Chroma420UtilityTests(unittest.TestCase):
    def test_posthoc_420_preserves_shape_and_bounded_range(self):
        rgb = torch.rand(2, 3, 8, 12)
        reconstructed = apply_posthoc_420_to_rgb(rgb)
        self.assertEqual(reconstructed.shape, rgb.shape)
        self.assertGreaterEqual(reconstructed.min().item(), 0.0)
        self.assertLessEqual(reconstructed.max().item(), 1.0)

    def test_posthoc_420_preserves_grayscale(self):
        gray = torch.rand(2, 1, 8, 12)
        rgb = gray.expand(-1, 3, -1, -1)
        reconstructed = apply_posthoc_420_to_rgb(rgb)
        self.assertTrue(torch.allclose(reconstructed, rgb, atol=1e-6))

    def test_downsample_rejects_odd_dimensions(self):
        with self.assertRaises(ValueError):
            downsample_chroma_420(torch.rand(1, 2, 7, 12))

    def test_scale_four_downsample_has_expected_shape(self):
        chroma = torch.rand(1, 2, 8, 12)
        self.assertEqual(downsample_chroma(chroma, scale=4).shape, (1, 2, 2, 3))

    def test_representation_sample_ratios(self):
        self.assertEqual(
            representation_sample_ratio(SimpleNamespace(experiment='posthoc420', chroma_scale=2)),
            0.5,
        )
        self.assertEqual(
            representation_sample_ratio(
                SimpleNamespace(experiment='neural420_early_chroma', chroma_scale=4)),
            0.375,
        )
        self.assertEqual(
            representation_sample_ratio(SimpleNamespace(experiment='rgb444', chroma_scale=2)),
            1.0,
        )
        self.assertEqual(
            representation_sample_ratio(SimpleNamespace(experiment='rgb_asym', chroma_scale=2)),
            1.0,
        )
        self.assertEqual(
            representation_sample_ratio(SimpleNamespace(experiment='rgb_es180_upper', chroma_scale=4)),
            1.0,
        )
        self.assertEqual(
            representation_sample_ratio(SimpleNamespace(experiment='yuv_es180_upper', chroma_scale=4)),
            0.375,
        )

    def test_reconstruction_restores_full_resolution(self):
        y = torch.rand(1, 1, 8, 12)
        cbcr_low = torch.rand(1, 2, 4, 6)
        reconstructed = reconstruct_rgb_from_420(y, cbcr_low)
        self.assertEqual(reconstructed.shape, (1, 3, 8, 12))

    def test_learned_reconstruction_requires_module(self):
        with self.assertRaises(ValueError):
            reconstruct_rgb_from_y_and_chroma(
                torch.rand(1, 1, 8, 12),
                torch.rand(1, 2, 4, 6),
                chroma_upsampler='learned',
            )

    def test_learned_residual_upsampler_restores_full_resolution(self):
        upsampler = LearnedChromaUpsampler(width=4, depth=1, residual=True)
        cbcr = upsampler(torch.rand(1, 2, 4, 6), target_size=(8, 12))
        self.assertEqual(cbcr.shape, (1, 2, 8, 12))
        self.assertGreaterEqual(cbcr.min().item(), 0.0)
        self.assertLessEqual(cbcr.max().item(), 1.0)

    def test_weighted_yuv_psnr_dbavg(self):
        self.assertEqual(weighted_yuv_psnr_dbavg(30, 40, 50), 33.75)

    def test_weighted_yuv_psnr_mse(self):
        self.assertAlmostEqual(weighted_yuv_psnr_mse(1e-3, 1e-3, 1e-3), 30.0, places=6)

    def test_temporal_rgb_error_diff_zero_for_constant_error(self):
        errors = [torch.ones(3, 4, 4), torch.ones(3, 4, 4)]
        self.assertEqual(temporal_rgb_error_diff_from_errors(errors), 0.0)

    def test_temporal_rgb_error_diff_nan_for_single_frame(self):
        value = temporal_rgb_error_diff_from_errors([torch.ones(3, 4, 4)])
        self.assertTrue(np.isnan(value))


class ChromaGeneratorTests(unittest.TestCase):
    def test_shared_trunk_output_shapes(self):
        model = ChromaGenerator('neural420_shared', **generator_kwargs())
        output = model(torch.rand(1, 4))
        self.assertEqual(output['y'].shape, (1, 1, 8, 12))
        self.assertEqual(output['cbcr_low'].shape, (1, 2, 4, 6))

    def test_split_branch_output_shapes(self):
        model = ChromaGenerator('neural420_split', **generator_kwargs())
        output = model(torch.rand(1, 4))
        self.assertEqual(output['y'].shape, (1, 1, 8, 12))
        self.assertEqual(output['cbcr_low'].shape, (1, 2, 4, 6))

    def test_learned_upsampler_is_attached(self):
        kwargs = generator_kwargs()
        kwargs['chroma_upsampler'] = 'learned'
        kwargs['learned_upsampler_width'] = 4
        kwargs['learned_upsampler_depth'] = 1
        kwargs['learned_upsampler_residual'] = True
        model = ChromaGenerator('neural420_shared_learned_up', **kwargs)
        self.assertIsInstance(model.learned_upsampler, LearnedChromaUpsampler)

    def test_early_chroma_output_shapes(self):
        kwargs = generator_kwargs()
        kwargs['chroma_scale'] = 4
        model = ChromaGenerator('neural420_early_chroma', **kwargs)
        output = model(torch.rand(1, 4))
        self.assertEqual(output['y'].shape, (1, 1, 8, 12))
        self.assertEqual(output['cbcr_low'].shape, (1, 2, 2, 3))

    def test_asymmetric_y_branch_uses_requested_width(self):
        kwargs = generator_kwargs()
        kwargs['y_branch_width'] = 3
        model = ChromaGenerator('neural420_asym_y', **kwargs)
        output = model(torch.rand(1, 4))
        self.assertEqual(model.y_adapter.out_channels, 3)
        self.assertEqual(output['y'].shape, (1, 1, 8, 12))
        self.assertEqual(output['cbcr_low'].shape, (1, 2, 4, 6))

    def test_narrower_asymmetric_y_branch_reduces_estimated_flops(self):
        wide_kwargs = generator_kwargs()
        wide_kwargs['y_branch_width'] = 4
        narrow_kwargs = generator_kwargs()
        narrow_kwargs['y_branch_width'] = 2
        wide = ChromaGenerator('neural420_asym_y', **wide_kwargs)
        narrow = ChromaGenerator('neural420_asym_y', **narrow_kwargs)
        embed_input = torch.rand(1, 4)
        args = predict_args('neural420_asym_y')
        self.assertLess(
            estimate_model_gflops(narrow, embed_input, args),
            estimate_model_gflops(wide, embed_input, args),
        )

    def test_rgb_asym_output_shape(self):
        kwargs = generator_kwargs()
        kwargs['rgb_branch_width'] = 2
        model = RGBAsymGenerator(**kwargs)
        output = model(torch.rand(1, 4))
        self.assertEqual(output.shape, (1, 3, 8, 12))

    def test_narrower_rgb_asym_branch_reduces_estimated_flops(self):
        wide_kwargs = generator_kwargs()
        wide_kwargs['rgb_branch_width'] = 4
        narrow_kwargs = generator_kwargs()
        narrow_kwargs['rgb_branch_width'] = 2
        wide = RGBAsymGenerator(**wide_kwargs)
        narrow = RGBAsymGenerator(**narrow_kwargs)
        embed_input = torch.rand(1, 4)
        args = predict_args('rgb_asym')
        self.assertLess(
            estimate_model_gflops(narrow, embed_input, args),
            estimate_model_gflops(wide, embed_input, args),
        )

    def test_rgb_early_split_upper_output_shape(self):
        kwargs = generator_kwargs()
        kwargs['stride_list'] = [2, 2, 2]
        kwargs['upper_branch_width'] = 2
        model = RGBEarlySplitUpperGenerator(**kwargs)
        output = model(torch.rand(1, 4))
        self.assertEqual(output.shape, (1, 3, 16, 24))

    def test_yuv_early_split_upper_output_shapes(self):
        kwargs = generator_kwargs()
        kwargs['stride_list'] = [2, 2, 2]
        kwargs['upper_branch_width'] = 2
        model = YUVEarlySplitUpperGenerator(**kwargs)
        output = model(torch.rand(1, 4))
        self.assertEqual(output['y'].shape, (1, 1, 16, 24))
        self.assertEqual(output['cbcr_low'].shape, (1, 2, 4, 6))

    def test_narrower_rgb_early_split_upper_reduces_estimated_flops(self):
        wide_kwargs = generator_kwargs()
        wide_kwargs['stride_list'] = [2, 2, 2]
        wide_kwargs['upper_branch_width'] = 4
        narrow_kwargs = generator_kwargs()
        narrow_kwargs['stride_list'] = [2, 2, 2]
        narrow_kwargs['upper_branch_width'] = 2
        wide = RGBEarlySplitUpperGenerator(**wide_kwargs)
        narrow = RGBEarlySplitUpperGenerator(**narrow_kwargs)
        embed_input = torch.rand(1, 4)
        args = predict_args('rgb_es180_upper')
        self.assertLess(
            estimate_model_gflops(narrow, embed_input, args),
            estimate_model_gflops(wide, embed_input, args),
        )

    def test_narrower_yuv_early_split_upper_reduces_estimated_flops(self):
        wide_kwargs = generator_kwargs()
        wide_kwargs['stride_list'] = [2, 2, 2]
        wide_kwargs['upper_branch_width'] = 4
        narrow_kwargs = generator_kwargs()
        narrow_kwargs['stride_list'] = [2, 2, 2]
        narrow_kwargs['upper_branch_width'] = 2
        wide = YUVEarlySplitUpperGenerator(**wide_kwargs)
        narrow = YUVEarlySplitUpperGenerator(**narrow_kwargs)
        embed_input = torch.rand(1, 4)
        args = predict_args('yuv_es180_upper')
        self.assertLess(
            estimate_model_gflops(narrow, embed_input, args),
            estimate_model_gflops(wide, embed_input, args),
        )

    def test_existing_experiments_still_instantiate(self):
        for experiment in [
                'rgb444', 'posthoc420', 'neural420_shared', 'neural420_asym_y',
                'rgb_es180_upper', 'yuv_es180_upper']:
            with self.subTest(experiment=experiment):
                model = build_model(trainer_args(experiment), embed_length=4)
                self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
