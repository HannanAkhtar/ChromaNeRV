# NeRV Generalization Experiments

## Purpose

This pipeline tests whether ChromaNeRV's luma-aware capacity allocation transfers
to the original NeRV decoder. Each UVG sequence is trained independently. The
primary matched comparisons are `rgbsplit_w8` versus `chroma_w8` and
`rgbsplit_w4` versus `chroma_w4`.

`full_rgb` predicts unrestricted full-resolution RGB. `full_ycbcr444` uses the
same full decoder but predicts full-resolution BT.709 YCbCr. RGBSplit restricts
the final RGB stage to width 8 or 4. ChromaNeRV uses the same shared trunk,
split, and final Y-branch width, while CbCr exits at the penultimate stage.
The ES180 variants remain available as historical aggressive extensions; they
are not part of this primary supplementary grid.

## Data And Architecture

The default sequence list is Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo,
ShakeNDry, and YachtRide. PNG is preferred over JPG/JPEG, names are naturally
sorted, and exactly the first 132 frames are selected. Main runs require
960x1920 model inputs and never resize implicitly.

Production UVG frames may be stored at either 960x1920 or 1080x1920. Frames
already at 960x1920 are used unchanged. Frames at exactly 1080x1920 receive the
same deterministic center crop used by the ChromaHNeRV experiments:

```text
top=60, left=0, height=960, width=1920
```

This removes 60 rows from the top and bottom, removes no columns, and performs
no interpolation. No other automatic crop is supported. Any other source
resolution fails unless `--allow_resize` is explicitly supplied for a
compatibility or debug run.

Each run records the source and target resolutions, preprocessing mode, crop
coordinates, and resize flag in `config.json` and `environment.json`. These
fields are included in the scientific configuration hash, preventing runs with
different spatial preprocessing from being resumed or treated as equivalent.

The 960x1920 NeRV-S preset is:

```text
embed=1.25_40, stem_dim_num=512_1, fc_hw_dim=8_16_26
expansion=1, reduction=2, lower_width=96, num_blocks=1
norm=none, act=swish, conv_type=conv, strides=5 3 2 2 2
single_res=true
```

Spatial progression is 8x16 -> 40x80 -> 120x240 -> 240x480 -> 480x960
-> 960x1920. The structural split is the output of the penultimate decoder
stage, 480x960 for this preset.

## Training And Losses

Final defaults are 300 epochs, batch size 1, Adam with learning rate `5e-4`,
betas `(0.5, 0.999)`, zero weight decay, cosine scheduling, 20% (60 epoch)
warmup, seed 1, and shuffled frame training. The reported checkpoint defaults
to the final epoch.

Full RGB and RGBSplit optimize RGB MSE. Full YCbCr444 optimizes equal MSE over
full-resolution Y, Cb, and Cr. ChromaNeRV optimizes:

```text
MSE(Y) + MSE(CbCr_half) + 0.1 * MSE(reconstructed_RGB)
```

Targets use full-range BT.709 and area chroma downsampling. Reconstruction uses
bilinear interpolation with `align_corners=False`, inverse full-range BT.709,
and RGB clamping to `[0, 1]`.

## Metrics

RGB PSNR, Y-PSNR, Cb-PSNR, and Cr-PSNR use one global sequence MSE and `MAX=1`.
RGB MS-SSIM is evaluated per frame with `data_range=1` and equally averaged.
YUV SSIM similarly averages component SSIM per frame before 6:1:1 weighting.
LPIPS uses AlexNet and RGB mapped to `[-1, 1]`; DISTS uses RGB `[0, 1]`.

The paper-facing weighted YUV PSNR is MSE-first:

```text
mse_611 = (6*mse_y + mse_cb + mse_cr) / 8
PSNR_611 = 10*log10(1/mse_611)
```

Do not substitute a weighted average of component PSNR values. That legacy
dB-average is retained only under the diagnostic name
`yuv_psnr_611_dbavg`.

VMAF uses FFmpeg's `libvmaf` filter, equal image counts and rates, and a shared
`yuv420p` conversion path. The complete command and JSON log are saved. The
default model is VMAF v0.6.1 unless `--vmaf_model_path` is supplied.

FID is computed once per configuration from the pooled UVG7 distribution:
924 reference images versus the corresponding 924 reconstructions. It is never
computed per sequence and then averaged.

Model complexity counts Linear, Conv2d, and ConvTranspose2d operations. One MAC
is two FLOPs. Color conversion, interpolation, PixelShuffle rearrangement,
metrics, data loading, and file encoding are excluded. FPS is model-only;
end-to-end reconstruction FPS is separately labeled.

The local legacy quantization path does not encode all codebook, shape, scale,
zero-point, and tensor-boundary overhead. `--rate_eval` therefore records rate
and BD-rate as unavailable rather than emitting an incompatible estimate.

## Automatic Persistence

`--output_root` is the local working root. It contains active checkpoints,
logs, caches, predictions, references, and temporary FID/VMAF inputs.
`--persistent_root` is the durable OneDrive root. It contains atomic copies of
manifests, final and best checkpoints, logs, metrics, frame lists, completion
markers, pooled FID, and aggregation results.

Predictions and references are local-only by default. Use
`--persist_predictions` only when their additional OneDrive storage cost is
intentional. The default `--persist_checkpoint_interval 10` copies a resumable
`model_latest.pth` every ten epochs. Set it to zero to disable intermediate
persistence.

Every durable write first targets `filename.tmp` in the destination directory,
is flushed and verified, and then replaces the destination with `os.replace`.
Checkpoint copies are size- and SHA-256-verified. `completion.json` is written
last and records the scientific configuration hash, selected checkpoint hash,
metrics hash, paths, and required artifacts. A persistent directory is never
treated as complete merely because it exists.

After an RDP disconnection, rerun the same command with `--resume`; a matching
local checkpoint is resumed and pending persistence is retried first. After
DeepFreeze or local disk loss, the launcher validates the persistent completion
marker and skips completed training. A valid periodic checkpoint is restored
when the run was not complete. If pooled-FID PNGs were not persisted, evaluation
is run automatically from the restored final checkpoint to regenerate them
locally. They are deleted after FID unless `--keep_predictions` is supplied.

Persistence retries default to three attempts with ten seconds between attempts.
After final failure, local data remains untouched and
`persistence_pending.json` is written. The launcher stops before the next job
unless `--continue_on_persistence_failure` is explicitly selected.

The persistent layout is:

```text
persistent_root/
  manifests/
  runs/<sequence>/<config>/
  results/
```

Omitting `--persistent_root` preserves the old local-only behavior and prints a
warning.

## Commands

Environment:

```bash
python scripts/check_nerv_generalization_environment.py
```

Dry run:

```bat
python scripts\run_nerv_generalization.py ^
  --data_root "C:\Users\b00101092\Documents\Chroma\UVG_extracted" ^
  --output_root "C:\Users\b00101092\Documents\Chroma\runs\nerv_generalization" ^
  --persistent_root "C:\Users\b00101092\OneDrive - aus.edu\persistent_storage\nerv_generalization" ^
  --dry_run
```

Smoke test:

```bat
python scripts\run_nerv_generalization.py ^
  --data_root "C:\Users\b00101092\Documents\Chroma\UVG_extracted" ^
  --output_root "C:\Users\b00101092\Documents\Chroma\runs\nerv_generalization_smoke" ^
  --persistent_root "C:\Users\b00101092\OneDrive - aus.edu\persistent_storage\nerv_generalization_smoke" ^
  --sequences Beauty ^
  --configs full_rgb,chroma_w8 ^
  --max_frames 2 ^
  --epochs 1 ^
  --smoke_test ^
  --skip_vmaf ^
  --skip_fid
```

One sequence:

```bat
python scripts\run_nerv_generalization.py ^
  --data_root "C:\Users\b00101092\Documents\Chroma\UVG_extracted" ^
  --output_root "C:\Users\b00101092\Documents\Chroma\runs\nerv_generalization" ^
  --persistent_root "C:\Users\b00101092\OneDrive - aus.edu\persistent_storage\nerv_generalization" ^
  --sequences Beauty ^
  --max_frames 132 ^
  --epochs 300 ^
  --checkpoint_policy final ^
  --resume
```

Full UVG7:

```bat
python scripts\run_nerv_generalization.py ^
  --data_root "C:\Users\b00101092\Documents\Chroma\UVG_extracted" ^
  --output_root "C:\Users\b00101092\Documents\Chroma\runs\nerv_generalization" ^
  --persistent_root "C:\Users\b00101092\OneDrive - aus.edu\persistent_storage\nerv_generalization" ^
  --max_frames 132 ^
  --epochs 300 ^
  --checkpoint_policy final ^
  --resume
```

Evaluation-only:

```bash
python scripts/run_nerv_generalization.py --data_root /path/to/UVG_extracted --output_root output/nerv_generalization --sequences Beauty --configs chroma_w8 --max_frames 132 --epochs 300 --eval_only
```

Aggregation:

```bat
python scripts\aggregate_nerv_generalization.py ^
  --results_root "C:\Users\b00101092\Documents\Chroma\runs\nerv_generalization" ^
  --persistent_root "C:\Users\b00101092\OneDrive - aus.edu\persistent_storage\nerv_generalization"
```

Persistence validation is performed automatically at launcher startup and
before every skip or restore decision. It can also be exercised with:

```bat
python -m unittest tests.test_nerv_persistence -v
```
