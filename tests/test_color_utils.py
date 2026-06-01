import unittest

import torch

from utils import rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709


class Bt709ColorConversionTests(unittest.TestCase):
    def test_reference_values(self):
        rgb = torch.tensor(
            [[[[1.0]], [[0.0]], [[0.0]]],
             [[[0.0]], [[1.0]], [[0.0]]],
             [[[0.0]], [[0.0]], [[1.0]]],
             [[[0.0]], [[0.0]], [[0.0]]],
             [[[1.0]], [[1.0]], [[1.0]]]]
        )
        expected = torch.tensor(
            [[[[0.2126]], [[0.385428]], [[1.0]]],
             [[[0.7152]], [[0.114572]], [[0.045847]]],
             [[[0.0722]], [[1.0]], [[0.454153]]],
             [[[0.0]], [[0.5]], [[0.5]]],
             [[[1.0]], [[0.5]], [[0.5]]]]
        )
        self.assertTrue(torch.allclose(rgb_to_ycbcr_bt709(rgb), expected, atol=1e-5))

    def test_shape_dtype_and_device_are_preserved(self):
        rgb = torch.rand(2, 3, 5, 7, dtype=torch.float64)
        ycbcr = rgb_to_ycbcr_bt709(rgb)
        self.assertEqual(ycbcr.shape, rgb.shape)
        self.assertEqual(ycbcr.dtype, rgb.dtype)
        self.assertEqual(ycbcr.device, rgb.device)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_device_is_preserved(self):
        rgb = torch.rand(2, 3, 5, 7, device='cuda')
        reconstructed = ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(rgb))
        self.assertEqual(reconstructed.device, rgb.device)
        self.assertLess((rgb - reconstructed).abs().max().item(), 1e-6)

    def test_rgb_to_ycbcr_is_bounded_for_bounded_rgb(self):
        ycbcr = rgb_to_ycbcr_bt709(torch.rand(4, 3, 8, 8))
        self.assertGreaterEqual(ycbcr.min().item(), 0.0)
        self.assertLessEqual(ycbcr.max().item(), 1.0)

    def test_round_trip_error_is_negligible(self):
        rgb = torch.rand(4, 3, 8, 8)
        reconstructed = ycbcr_to_rgb_bt709(rgb_to_ycbcr_bt709(rgb))
        self.assertLess((rgb - reconstructed).abs().max().item(), 1e-6)

    def test_invalid_channel_count_is_rejected(self):
        with self.assertRaises(ValueError):
            rgb_to_ycbcr_bt709(torch.rand(2, 4, 8, 8))
        with self.assertRaises(ValueError):
            ycbcr_to_rgb_bt709(torch.rand(2, 1, 8, 8))


if __name__ == '__main__':
    unittest.main()
