# ChromaNeRV Ablation v2 Implementation Specification

## Purpose

This document extends the current ChromaNeRV implementation after the initial 4:2:0 experiments.

Current findings:

1. Post-hoc 4:2:0 is almost lossless on Bunny.
2. `neural420_shared` improves or preserves luma better than RGB, but loses chroma accuracy.
3. `neural420_split` gives only a small FPS gain and a larger quality drop.
4. Reducing the Y branch too aggressively may hurt PSNR because Y carries most edges and structure.
5. The next step is to improve low-resolution chroma modeling first, and only then explore compute-saving variants.

The coding agent should modify the existing ChromaNeRV files and add ablation support for:

1. Tuning neural 4:2:0 without reducing Y.
2. Adding a learned chroma upsampler.
3. Trying earlier chroma stopping / lower chroma resolution, without claiming FLOP savings yet.
4. Carefully testing Y-only branch width reduction for actual compute savings.

All new results must be stored in a new CSV:

```text
results/chroma420_bunny_ablation_v2.csv
```

Do not overwrite the old CSV:

```text
results/chroma420_bunny_nervs.csv
```

---

# Existing Files

The current codebase contains:

```text
train_chroma_nerv.py
model_chroma_nerv.py
train_nerv.py
model_nerv.py
utils.py
tests/test_chroma_nerv.py
tests/test_color_utils.py
```

The new work should mainly modify:

```text
train_chroma_nerv.py
model_chroma_nerv.py
tests/test_chroma_nerv.py
```

Only minor backward-compatible changes should be made to:

```text
utils.py
```

Do not break the existing `train_nerv.py`.

---

# Required Global Fixes Before New Ablations

## 1. Fix PyTorch 2.6+ checkpoint loading

In `train_chroma_nerv.py`, update checkpoint loading:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
```

Use this only for trusted local checkpoints.

## 2. Fix output_sample_ratio for posthoc420

Currently posthoc420 may be logged with:

```text
output_sample_ratio = 1.0
```

This is wrong for representation accounting.

Change:

```python
'output_sample_ratio': 0.5 if args.experiment in EXPERIMENTS_NEURAL_420 else 1.0
```

to:

```python
'output_sample_ratio': 0.5 if args.experiment in EXPERIMENTS_420 else 1.0
```

Important interpretation:

- `posthoc420` has representation sample ratio 0.5.
- But it does not reduce neural model FLOPs or parameters.

## 3. Add optional FLOP estimation

Add approximate architecture-level FLOP estimation and log it to CSV.

New CSV column:

```text
estimated_gflops
```

If exact FLOPs are hard to implement, use formula-based estimates based on the NeRV block settings.

For NeRV-S baseline:

```text
fc_hw_dim = 9_16_26
strides = [5,2,2,2,2]
lower_width = 96
expansion = 1
reduction = 2
resolution = 720x1280
```

Known baseline:

```text
~201.88 GFLOPs/frame
```

If `estimated_gflops` is not implemented immediately, leave it blank or `NaN`, but keep the column.

---

# New CLI Arguments

Add the following arguments to `train_chroma_nerv.py`.

```python
parser.add_argument("--lambda_y", type=float, default=1.0)
parser.add_argument("--lambda_c", type=float, default=1.0)
parser.add_argument("--lambda_rgb", type=float, default=0.0)
```

Already present, but ensure they are respected for all neural 4:2:0 variants.

Add:

```python
parser.add_argument(
    "--chroma_upsampler",
    default="bilinear",
    choices=["nearest", "bilinear", "bicubic", "learned"],
)
```

This is different from `--chroma_upsample`, but for backward compatibility either:

- replace `--chroma_upsample` with `--chroma_upsampler`, or
- allow both, with `--chroma_upsampler` overriding.

Recommended:

Keep the old flag and add the new one:

```python
parser.add_argument(
    "--chroma_upsampler",
    default=None,
    choices=[None, "nearest", "bilinear", "bicubic", "learned"],
)
```

Then resolve:

```python
if args.chroma_upsampler is None:
    args.chroma_upsampler = args.chroma_upsample
```

Add learned upsampler parameters:

```python
parser.add_argument("--learned_upsampler_width", type=int, default=16)
parser.add_argument("--learned_upsampler_depth", type=int, default=2)
parser.add_argument("--learned_upsampler_residual", action="store_true")
```

Add earlier chroma stopping / chroma-resolution parameters:

```python
parser.add_argument(
    "--chroma_scale",
    type=int,
    default=2,
    choices=[2, 4],
    help="Chroma downsampling factor relative to full resolution. 2 means 4:2:0, 4 means more aggressive low-res chroma."
)
```

Interpretation:

```text
chroma_scale = 2:
    Cb/Cr = H/2 x W/2

chroma_scale = 4:
    Cb/Cr = H/4 x W/4
```

For 720x1280:

```text
scale=2 -> 360x640
scale=4 -> 180x320
```

Add Y-branch width reduction:

```python
parser.add_argument("--y_branch_width", type=int, default=96)
```

Add optional chroma branch width:

```python
parser.add_argument("--chroma_branch_width", type=int, default=96)
```

This may be unused initially, but include it in the CSV for future work.

Add CSV run grouping:

```python
parser.add_argument("--ablation_group", default="")
```

Examples:

```text
loss_sweep
learned_upsampler
early_chroma
y_width_sweep
```

---

# New Experiment Names

Extend:

```python
choices=[
    "rgb444",
    "ycbcr444",
    "posthoc420",
    "neural420_shared",
    "neural420_split",
]
```

to:

```python
choices=[
    "rgb444",
    "ycbcr444",
    "posthoc420",
    "neural420_shared",
    "neural420_split",
    "neural420_shared_learned_up",
    "neural420_early_chroma",
    "neural420_asym_y",
]
```

Meaning:

## `neural420_shared`

Existing model.

Full NeRV trunk to 720x1280, predicts:

```text
Y full-res
CbCr low-res
```

Use for loss tuning.

## `neural420_shared_learned_up`

Same prediction structure as `neural420_shared`, but evaluation/reconstruction uses learned chroma upsampling instead of bilinear interpolation.

## `neural420_early_chroma`

Predict chroma at a lower resolution than normal 4:2:0.

Recommended first setting:

```text
chroma_scale = 4
CbCr = H/4 x W/4 = 180x320
Y = H x W = 720x1280
```

Do not claim FLOP savings yet unless the code actually avoids corresponding computation.

## `neural420_asym_y`

Split model where Y branch has configurable width:

```text
y_branch_width = 96, 80, 64, 48, 32
```

This is the first compute-saving variant.

---

# Model Changes in model_chroma_nerv.py

## 1. Generalize chroma resize functions

Current 4:2:0 functions assume scale 2.

Add generalized functions:

```python
def downsample_chroma(chroma, scale=2, mode="area"):
    height, width = chroma.shape[-2:]
    if height % scale or width % scale:
        raise ValueError(...)
    return resize_chroma(chroma, (height // scale, width // scale), mode)
```

```python
def reconstruct_rgb_from_y_and_chroma(
    y,
    cbcr_low,
    chroma_upsampler="bilinear",
    learned_upsampler=None,
    clamp=True,
):
    ...
```

If `chroma_upsampler == "learned"`:

```python
cbcr_full = learned_upsampler(cbcr_low, target_size=y.shape[-2:])
```

Otherwise use interpolation.

Keep backward-compatible wrappers:

```python
downsample_chroma_420(...)
reconstruct_rgb_from_420(...)
apply_posthoc_420_to_rgb(...)
```

---

## 2. Add LearnedChromaUpsampler

Add class:

```python
class LearnedChromaUpsampler(nn.Module):
    def __init__(self, width=16, depth=2, residual=False):
        super().__init__()
        ...
```

Recommended implementation:

Input:

```text
CbCr low-res [B,2,Hc,Wc]
```

Process:

1. Upsample to target size using bilinear.
2. Pass through small CNN.
3. Output full-res CbCr.

Pseudo-code:

```python
class LearnedChromaUpsampler(nn.Module):
    def __init__(self, width=16, depth=2, residual=False):
        super().__init__()
        layers = []
        in_ch = 2
        for i in range(depth):
            layers.append(nn.Conv2d(in_ch, width, 3, 1, 1))
            layers.append(nn.SiLU(inplace=True))
            in_ch = width
        layers.append(nn.Conv2d(width, 2, 3, 1, 1))
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, cbcr_low, target_size):
        base = F.interpolate(cbcr_low, size=target_size, mode="bilinear", align_corners=False)
        correction = self.net(base)
        if self.residual:
            return (base + correction).clamp(0, 1)
        return torch.sigmoid(correction)
```

Alternative:

If using normalized CbCr in [0,1], a residual correction may be more stable:

```python
return (base + 0.1 * torch.tanh(correction)).clamp(0,1)
```

Preferred first version:

```text
learned_upsampler_residual = True
residual scale = 0.1
```

---

## 3. Modify ChromaGenerator

The current `ChromaGenerator` supports:

```text
neural420_shared
neural420_split
```

Extend it to support:

```text
neural420_shared_learned_up
neural420_early_chroma
neural420_asym_y
```

### For neural420_shared_learned_up

Same architecture as `neural420_shared`, but include:

```python
self.learned_upsampler = LearnedChromaUpsampler(...)
```

if `chroma_upsampler == "learned"`.

### For neural420_early_chroma

Goal:

```text
Y: full resolution
CbCr: H/4 x W/4 if chroma_scale=4
```

Simplest implementation:

Use full shared trunk, then resize CbCr prediction to target chroma resolution.

This does not claim compute savings.

Pseudo:

```python
cbcr_logits = self.cbcr_head(features_full)
cbcr_low = resize_chroma(cbcr_logits, (H//scale, W//scale), mode="area")
```

Better implementation later:

Generate CbCr directly from earlier feature map.

But for this ablation, the main question is quality sensitivity to chroma resolution, not compute.

### For neural420_asym_y

Architecture:

```text
shared trunk up to 360x640
    ├── chroma head at 360x640 or lower
    └── Y adapter 96 -> y_branch_width
        final upsampling block
        Y head
```

Important:

The final upsampling block should use `y_branch_width`.

Pseudo:

```python
self.y_adapter = nn.Conv2d(shared_ngf, y_branch_width, 1, 1)
self.y_layers = build_stage(
    ngf=y_branch_width,
    new_ngf=y_branch_width,
    stride=2,
)
self.y_head = nn.Conv2d(y_branch_width, 1, 1)
```

The expensive final block changes from:

```text
96 -> 384 before PixelShuffle
```

to:

```text
y_branch_width -> y_branch_width*4 before PixelShuffle
```

If y_branch_width=64:

```text
64 -> 256
```

This should reduce FLOPs significantly.

Be careful:

This is no longer exactly the same as the previous split model, because the final Y block input and output channels are reduced.

---

# Training Loss Updates

For all neural 4:2:0 variants:

```python
loss_y = MSE(y_pred, y_target)
loss_c = MSE(cbcr_pred_low, cbcr_target_low)
loss_rgb = MSE(rgb_pred, rgb_target)
loss = lambda_y*loss_y + lambda_c*loss_c + lambda_rgb*loss_rgb
```

When `chroma_scale=4`, target low-res CbCr is:

```text
H/4 x W/4
```

not H/2 x W/2.

Use:

```python
cbcr_target_low = downsample_chroma(cbcr_target, scale=args.chroma_scale, mode=args.chroma_downsample)
```

---

# Evaluation Updates

For all variants, compute:

```text
RGB PSNR
RGB MS-SSIM
PSNR-Y
PSNR-Cb
PSNR-Cr
model_fps
end_to_end_fps
params_M
checkpoint_size_MB
estimated_gflops
```

Add CSV columns:

```text
ablation_group
lambda_y
lambda_c
lambda_rgb
chroma_scale
chroma_upsampler
learned_upsampler_width
learned_upsampler_depth
learned_upsampler_residual
y_branch_width
chroma_branch_width
estimated_gflops
```

CSV file:

```text
results/chroma420_bunny_ablation_v2.csv
```

Do not overwrite automatically.

---

# Ablation Plan

## Group 1: Tune neural 4:2:0 without reducing Y

Goal:

Recover chroma accuracy while keeping PSNR-Y advantage.

Run:

```text
neural420_shared
lambda_y = 1.0
lambda_rgb = 0.0
lambda_c ∈ {1.0, 1.5, 2.0, 3.0}
```

Then:

```text
neural420_shared
lambda_y = 1.0
lambda_c = best from previous sweep
lambda_rgb ∈ {0.05, 0.1}
```

Success:

- RGB PSNR close to posthoc420
- MS-SSIM close to posthoc420
- PSNR-Y remains strong
- PSNR-Cb/Cr recover

---

## Group 2: Learned chroma upsampler

Goal:

Reduce color artifacts and recover RGB PSNR/MS-SSIM.

Start with best loss weights from Group 1.

Run:

```text
neural420_shared_learned_up
chroma_upsampler = learned
learned_upsampler_width ∈ {8, 16, 32}
learned_upsampler_depth = 2
learned_upsampler_residual = True
```

Optional:

```text
depth ∈ {1,2,3}
```

Success:

- Better RGB PSNR than bilinear neural420_shared
- Better PSNR-Cb/Cr
- Minimal FPS penalty

---

## Group 3: Earlier chroma stopping / more aggressive chroma reduction

Goal:

Measure how much chroma resolution can be reduced.

Do not claim compute savings unless implementation actually avoids computation.

Run:

```text
neural420_early_chroma
chroma_scale = 4
CbCr = H/4 x W/4
```

Use:

```text
best lambda_c
best lambda_rgb
bilinear upsampling first
learned upsampling second if needed
```

Success:

- Quality drop is bounded and understandable.
- Establishes chroma-resolution sensitivity.

Expected:

- RGB PSNR likely drops more than 4:2:0.
- PSNR-Y should remain strong.
- PSNR-Cb/Cr will drop.

This is useful as an ablation even if not final.

---

## Group 4: Careful Y-only width reduction

Goal:

Test real compute savings while protecting luma.

Run:

```text
neural420_asym_y
y_branch_width ∈ {96, 80, 64, 48, 32}
chroma_scale = 2
best lambda_c
best lambda_rgb
```

Recommended order:

```text
96 baseline first
80
64
48
32 only if 48 is acceptable
```

Success:

- Meaningful GFLOP/FPS reduction.
- RGB PSNR loss remains acceptable.
- PSNR-Y does not collapse.

Do not use a model if it saves compute but destroys PSNR-Y.

---

# Colab Commands

The coding agent should produce a helper notebook cell or shell script to run ablations.

Use Colab-safe Python variables, not shell variables.

Example checkpoint assignment:

```python
import glob

rgb_ckpt = glob.glob("output/**/*rgb_nervs_e300*/model_val_best.pth", recursive=True)[0]
ycbcr_ckpt = glob.glob("output/**/*ycbcr_nervs_e300*/model_val_best.pth", recursive=True)[0]
```

For newly trained models:

```python
shared_ckpt = glob.glob("output/chroma420_ablation_v2/bunny/<run_name>/model_val_best.pth")[0]
```

---

# Required New Commands

## 1. Loss sweep

```bash
python train_chroma_nerv.py -e 300 \
  --experiment neural420_shared \
  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
  --outf output/chroma420_ablation_v2 --embed 1.25_40 --stem_dim_num 512_1 \
  --reduction 2 --fc_hw_dim 9_16_26 --expansion 1 \
  --single_res --loss_type L2 --warmup 0.2 --lr_type cosine \
  --strides 5 2 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish \
  --lambda_y 1.0 --lambda_c <LC> --lambda_rgb <LRGB> \
  --chroma_downsample area --chroma_upsample bilinear \
  --chroma_scale 2 \
  --ablation_group loss_sweep \
  --run_name neural420_shared_lc<LC>_lrgb<LRGB> \
  --results_csv results/chroma420_bunny_ablation_v2.csv \
  --visual_frames 3 \
  --fps_repeats 30 \
  --overwrite
```

## 2. Learned upsampler

```bash
python train_chroma_nerv.py -e 300 \
  --experiment neural420_shared_learned_up \
  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
  --outf output/chroma420_ablation_v2 --embed 1.25_40 --stem_dim_num 512_1 \
  --reduction 2 --fc_hw_dim 9_16_26 --expansion 1 \
  --single_res --loss_type L2 --warmup 0.2 --lr_type cosine \
  --strides 5 2 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish \
  --lambda_y 1.0 --lambda_c <BEST_LC> --lambda_rgb <BEST_LRGB> \
  --chroma_downsample area \
  --chroma_upsampler learned \
  --learned_upsampler_width <WIDTH> \
  --learned_upsampler_depth 2 \
  --learned_upsampler_residual \
  --chroma_scale 2 \
  --ablation_group learned_upsampler \
  --run_name neural420_learnedup_w<WIDTH> \
  --results_csv results/chroma420_bunny_ablation_v2.csv \
  --visual_frames 3 \
  --fps_repeats 30 \
  --overwrite
```

## 3. Earlier chroma / chroma scale 4

```bash
python train_chroma_nerv.py -e 300 \
  --experiment neural420_early_chroma \
  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
  --outf output/chroma420_ablation_v2 --embed 1.25_40 --stem_dim_num 512_1 \
  --reduction 2 --fc_hw_dim 9_16_26 --expansion 1 \
  --single_res --loss_type L2 --warmup 0.2 --lr_type cosine \
  --strides 5 2 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish \
  --lambda_y 1.0 --lambda_c <BEST_LC> --lambda_rgb <BEST_LRGB> \
  --chroma_downsample area --chroma_upsample bilinear \
  --chroma_scale 4 \
  --ablation_group early_chroma \
  --run_name neural420_chromascale4 \
  --results_csv results/chroma420_bunny_ablation_v2.csv \
  --visual_frames 3 \
  --fps_repeats 30 \
  --overwrite
```

## 4. Y-only width sweep

```bash
python train_chroma_nerv.py -e 300 \
  --experiment neural420_asym_y \
  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
  --outf output/chroma420_ablation_v2 --embed 1.25_40 --stem_dim_num 512_1 \
  --reduction 2 --fc_hw_dim 9_16_26 --expansion 1 \
  --single_res --loss_type L2 --warmup 0.2 --lr_type cosine \
  --strides 5 2 2 2 2 --conv_type conv \
  -b 1 --lr 0.0005 --norm none --act swish \
  --lambda_y 1.0 --lambda_c <BEST_LC> --lambda_rgb <BEST_LRGB> \
  --chroma_downsample area --chroma_upsample bilinear \
  --chroma_scale 2 \
  --y_branch_width <YWIDTH> \
  --ablation_group y_width_sweep \
  --run_name neural420_asymy_w<YWIDTH> \
  --results_csv results/chroma420_bunny_ablation_v2.csv \
  --visual_frames 3 \
  --fps_repeats 30 \
  --overwrite
```

---

# Recommended Run Order

Do not run everything at once.

## Step 1: Unit tests

```bash
python -m unittest discover -s tests -v
```

## Step 2: Debug small runs

Run each new experiment for 2 epochs with `--debug`:

```text
neural420_shared with lambda_c=2
neural420_shared_learned_up
neural420_early_chroma
neural420_asym_y with y_branch_width=64
```

## Step 3: Loss sweep

Run:

```text
lambda_c = 1.5, 2.0, 3.0
```

Compare to the existing lambda_c=1.0 result.

## Step 4: RGB consistency

Use best lambda_c:

```text
lambda_rgb = 0.05, 0.1
```

## Step 5: Learned upsampler

Use best weights:

```text
width = 8, 16, 32
```

## Step 6: Earlier chroma

Run:

```text
chroma_scale = 4
```

## Step 7: Y width sweep

Only after quality is acceptable:

```text
y_branch_width = 80, 64, 48
```

Maybe 32 if 48 works.

---

# Final Analysis Script

The coding agent should add or provide a notebook cell:

```python
import pandas as pd

csv_path = "results/chroma420_bunny_ablation_v2.csv"
df = pd.read_csv(csv_path)

cols = [
    "ablation_group",
    "run_name",
    "experiment",
    "lambda_c",
    "lambda_rgb",
    "chroma_scale",
    "chroma_upsampler",
    "learned_upsampler_width",
    "y_branch_width",
    "params_M",
    "estimated_gflops",
    "model_fps",
    "end_to_end_fps",
    "rgb_psnr",
    "rgb_ms_ssim",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
]

display(df[cols])

df[cols].to_csv("results/chroma420_bunny_ablation_v2_summary.csv", index=False)
```

Also compute deltas relative to:

```text
posthoc420_from_rgb_nervs
```

and optionally relative to:

```text
rgb444_nervs_eval_trained
```

---

# Acceptance Criteria

The implementation is accepted when:

1. Existing experiments still run:
   - rgb444
   - ycbcr444
   - posthoc420
   - neural420_shared
   - neural420_split

2. New experiments run:
   - neural420_shared_learned_up
   - neural420_early_chroma
   - neural420_asym_y

3. New arguments are logged to CSV:
   - ablation_group
   - lambda_c
   - lambda_rgb
   - chroma_scale
   - chroma_upsampler
   - learned_upsampler_width
   - learned_upsampler_depth
   - y_branch_width
   - estimated_gflops

4. Results are written to:

```text
results/chroma420_bunny_ablation_v2.csv
```

5. Visual outputs are saved.

6. Unit tests pass.

7. No run writes `random_initialization` unless intentionally evaluating an untrained model.

8. All checkpoint loading works under PyTorch 2.6+.

---

# Scientific Interpretation After Ablations

The expected conclusions should be evaluated as follows.

## Loss tuning

If higher `lambda_c` improves RGB PSNR and PSNR-Cb/Cr with only minor PSNR-Y reduction, use that setting for all future runs.

## RGB consistency

If `lambda_rgb > 0` improves RGB PSNR/MS-SSIM, include it in the final method.

## Learned upsampler

If learned upsampling improves RGB PSNR or visual color quality, it becomes part of the stronger ChromaNeRV variant.

## Earlier chroma

If chroma_scale=4 has only small quality loss, it supports stronger chroma reduction.

If quality collapses, stay with chroma_scale=2.

## Y-width sweep

If y_branch_width=64 or 48 gives large FLOP/FPS gains with tolerable quality loss, it becomes the efficient ChromaNeRV variant.

If PSNR-Y drops too much, do not use width reduction and focus on storage/precision savings instead.

---

# Main Research Logic

Do not frame ChromaNeRV as simply reducing chroma after full decoding.

The intended logic is:

```text
1. Low-resolution chroma is perceptually acceptable.
2. Neural training can adapt to low-resolution chroma targets.
3. Loss and upsampling design recover most chroma artifacts.
4. Once quality is stable, compute/storage reductions can be explored via branch width and precision.
```

Protect Y first. Reduce chroma more aggressively before weakening the luma path.
