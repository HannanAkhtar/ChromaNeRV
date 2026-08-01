# ChromaNeRV: NeRV Generalization Supplement

## Anonymous supplementary-code notice

This anonymous repository accompanies the paper *ChromaNeRV: Luma-Chroma Capacity Allocation for Efficient Neural Video Representation*. It contains only the NeRV generalization experiments. The main-paper HNeRV training pipeline, model budgets, quantization experiments, rate-distortion curves, and BD-rate results are not included.

## Overview

The release compares six independently trained NeRV decoders on Bunny and seven UVG sequences. Each model uses the first 132 frames, seed 1, 300 epochs, batch size 1, and final-checkpoint evaluation. Final quality averages give equal weight to all eight videos.

## NeRV ChromaNeRV method

ChromaNeRV retains a shared NeRV trunk, then uses a narrow full-resolution Y branch and a half-resolution CbCr head. Chroma targets use area downsampling; predicted chroma uses bilinear upsampling before inverse full-range BT.709 conversion to RGB.

## Experimental configurations

`full_rgb` is the standard RGB decoder. `full_ycbcr444` has the same architecture and computational path but predicts full-resolution YCbCr as a color-space control. `rgbsplit_w8` and `rgbsplit_w4` restrict the final high-resolution RGB pathway. `chroma_w8` and `chroma_w4` use the luma-chroma allocation above with branch width 8 or 4.

## Repository structure

Core model and training code is at the root. Machine-readable protocols are in `configs/supplementary`, entry points are in `scripts`, shared metrics are in `metrics`, tiny test data is in `tests/fixtures`, and sanitized result metadata is in `results/supplementary`.

## Installation

The reference environment is Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, and a CUDA 12.4-compatible PyTorch build. Install PyTorch for the target CPU/CUDA platform, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/validate_release.py --check_environment
```

Conda users may instead use `conda env create -f environment.yml`.

## Optional metric dependencies

LPIPS, DISTS, FID/KID, FFmpeg with libvmaf, plotting, and Excel export are optional. Core imports do not require them. Install only the packages needed for the selected metrics; unavailable optional metrics report an actionable error or remain absent from output.

## Dataset preparation

Datasets are not distributed. Bunny may contain frames directly; UVG uses sequence subdirectories:

```text
data/bunny/000000.png
data/uvg/Beauty/000000.png
data/uvg/Bosphorus/000000.png
... HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide
```

Accepted suffixes are PNG, JPG/JPEG, BMP, TIFF, and WebP. Frames are naturally sorted by filename and the first 132 are selected. Paper runs require at least 132 frames. Bunny targets 720x1280 without resizing. UVG targets 960x1920 using a deterministic vertical center crop from 1080x1920. Normal runs never resize implicitly; `--allow_resize` is for smoke/debug use only.

## Dataset validation

```bash
python scripts/validate_release.py --check_data --bunny_root /path/to/bunny --uvg_root /path/to/uvg
```

The validator prints sequence discovery, selected count, first/last frame, source and target resolution, and preprocessing.

## Quick CPU smoke test

```bash
python scripts/run_nerv_generalization.py --config configs/supplementary/smoke_test.json --data_root tests/fixtures/tiny_video --output_root output/smoke_test --device cpu --smoke_test --skip_vmaf --skip_fid
```

The tiny grid exercises RGB, YCbCr444, RGBSplit, ChromaNeRV, training, checkpointing, and metric serialization. Use the evaluation-only command below to verify checkpoint reload.

## Reproducing Bunny experiments

```bash
python scripts/run_bunny_experiments.py --config configs/supplementary/nerv_bunny.json --data_root /path/to/bunny --output_root output/nerv_bunny --device cuda --resume
```

This schedules 1 sequence x 6 configurations. The supplied Bunny protocol uses `(lambda_y, lambda_c, lambda_rgb)=(1,3,0.1)` for ChromaNeRV; no completed Bunny config was available in the local checkout to independently confirm the chroma weight.

## Reproducing UVG7 experiments

```bash
python scripts/run_nerv_generalization.py --config configs/supplementary/nerv_uvg7.json --data_root /path/to/uvg --output_root output/nerv_uvg7 --device cuda --resume
```

This schedules 7 sequences x 6 configurations. UVG ChromaNeRV uses `(1,1,0.1)`.

## Running a selected sequence/configuration

```bash
python scripts/run_nerv_generalization.py --config configs/supplementary/nerv_uvg7.json --data_root /path/to/uvg --output_root output/nerv_uvg7 --sequences Beauty --configs chroma_w8 --device cuda
```

Add `--dry_run` to inspect the resolved grid without training. Existing scientifically mismatched run configs are rejected.

## Evaluation-only workflow

```bash
python scripts/evaluate_checkpoint.py --config output/nerv_uvg7/Beauty/chroma_w8/config.json --checkpoint output/nerv_uvg7/Beauty/chroma_w8/model_final.pth --data_root /path/to/uvg --output_root output/evaluation --device cuda
```

This entry point uses the training runner's metric implementations. Higher is better for PSNR, SSIM, VMAF, and VMAF-NEG. Lower is better for LPIPS, DISTS, temporal error, FID, and KID. FID/KID are secondary distributional measures; FPS is hardware-dependent; parameters and GFLOPs are architecture measurements.

## Aggregating Bunny and UVG7

```bash
python scripts/aggregate_supplementary_results.py --bunny_root output/nerv_bunny --uvg_root output/nerv_uvg7 --output_root results/supplementary
```

The command requires 48 unique jobs and averages sequence-level quality metrics equally over Bunny and UVG7. Pooled distributional metrics must be reported separately and are not arithmetic per-sequence averages.

## Reproducing supplementary tables

Aggregation writes full-precision CSVs and rounded LaTeX files for absolute results, matched ChromaNeRV-minus-RGBSplit controls, per-sequence controls, and Full-YCbCr444-minus-Full-RGB controls. No NeRV RD or BD-rate table is generated.

## Output files and directory structure

Each run writes `config.json`, commands and environment metadata, logs, final checkpoint, `eval_metrics.json`, `per_frame_metrics.csv`, and optional prediction/reference images. Grid manifests live under the selected output root. Local output works without persistent storage.

## Reference results for sanity checking

The local checkout did not contain the complete final 48-run result set, so this release does not invent numerical rows. `results/supplementary/manifest.json` records that limitation. The protocol notes supplied for release preparation suggest W8 should substantially reduce compute with small weighted-YUV and VMAF changes, and should usually outperform matched RGBSplit-W8; these are expectations to check against reproduced outputs, not bundled measurements.

## Reproducibility notes

Configs preserve separate Bunny and UVG spatial presets, architecture, frame selection, losses, optimizer, cosine schedule, warm-up, seed, and final-checkpoint policy. RGB conversion is full-range BT.709. A model is trained separately for every sequence/configuration pair.

## Known limitations

Datasets and checkpoints are excluded. Optional perceptual metrics can vary with dependency versions; VMAF requires an FFmpeg build with libvmaf. Pooled UVG7 FID/KID requires retained images. CPU/GPU timing is not portable. This repository provides no NeRV rate experiment or main-paper rate claim.

## Upstream NeRV attribution

This work is derived from the original NeRV implementation and paper, *NeRV: Neural Representations for Videos*. See `THIRD_PARTY_NOTICES.md` for attribution and dependency notices.

## Citation placeholder

```bibtex
@inproceedings{anonymous2027chromanerv,
  title={ChromaNeRV: Luma--Chroma Capacity Allocation for Efficient Neural Video Representation},
  author={Anonymous},
  booktitle={Under Review},
  year={2027}
}
```

Please cite the upstream NeRV paper separately when using its implementation.

## License

The supplementary modifications are released under the MIT License in `LICENSE`. Third-party components remain subject to their own terms; see `THIRD_PARTY_NOTICES.md`.
