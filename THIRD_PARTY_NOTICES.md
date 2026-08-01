# Third-Party Notices

This repository is derived from the original NeRV research implementation accompanying *NeRV: Neural Representations for Videos*. The upstream repository was distributed under the MIT License. Upstream copyright and citation information should be retained when redistributing derived source.

The project calls or optionally imports the following external implementations; they are not vendored in this archive:

| Component | Use | License/terms |
|---|---|---|
| PyTorch and torchvision | Model training and image features | BSD-style licenses; consult each installed package |
| pytorch-msssim | MS-SSIM | MIT |
| LPIPS | Learned perceptual metric | BSD-2-Clause |
| DISTS | Learned perceptual metric | Consult the installed upstream package; no license is asserted here |
| FFmpeg and libvmaf | VMAF/VMAF-NEG execution | Consult the exact binaries and their build-time license terms |
| torchmetrics / torch-fidelity | Optional FID and KID | Consult the installed package license |
| NumPy, SciPy, scikit-image, pandas, Pillow, tqdm | Data and evaluation utilities | Consult the installed package licenses |

No third-party metric model weights or dataset frames are included. Names of upstream projects are used only for attribution and dependency identification.

Upstream paper citation:

```bibtex
@inproceedings{chen2021nerv,
  title={NeRV: Neural Representations for Videos},
  author={Hao Chen and Bo He and Hanyu Wang and Yixuan Ren and Ser-Nam Lim and Abhinav Shrivastava},
  booktitle={NeurIPS},
  year={2021}
}
```
