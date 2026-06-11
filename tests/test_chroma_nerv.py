import unittest
from types import SimpleNamespace

import torch

from model_chroma_nerv import (
    ChromaGenerator,
    LearnedChromaUpsampler,
    apply_posthoc_420_to_rgb,
    downsample_chroma,
    downsample_chroma_420,
    reconstruct_rgb_from_y_and_chroma,
    reconstruct_rgb_from_420,
)
from train_chroma_nerv import estimate_model_gflops, representation_sample_ratio


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


if __name__ == '__main__':
    unittest.main()
