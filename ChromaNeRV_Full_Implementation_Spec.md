
# ChromaNeRV Design and Implementation Specification
Version 1.0

## Objective

Implement three ChromaNeRV experiments on top of the existing NeRV codebase:

1. Experiment A: Post-Hoc 4:2:0 Baseline
2. Experiment B: Neural 4:2:0 Shared Trunk
3. Experiment C: Neural 4:2:0 Split Branch Architecture

The implementation must be isolated from the existing Stage-1 code and should use:

- train_chroma_nerv.py
- model_chroma_nerv.py

The existing train_nerv.py and model_nerv.py must remain functional.

---

# Background

Stage 1 demonstrated:

RGB NeRV-S:
- RGB PSNR = 32.86
- MS-SSIM = 0.9506
- PSNR-Y = 33.43

YCbCr 4:4:4 NeRV-S:
- RGB PSNR = 32.76
- MS-SSIM = 0.9492
- PSNR-Y = 33.56

Conclusion:

Color-space conversion alone does not significantly improve RGB quality.
The main ChromaNeRV contribution should therefore come from:

- reduced chroma resolution
- reduced chroma compute
- reduced chroma capacity
- later reduced chroma precision

---

# Experiment A
# Post-Hoc 4:2:0 Baseline

Purpose:

Measure the quality loss caused by ordinary codec-style chroma subsampling.

No architecture changes are allowed.

Pipeline:

Model Output
    ↓
RGB
    ↓
YCbCr
    ↓
Cb/Cr Downsample
    ↓
Cb/Cr Upsample
    ↓
RGB
    ↓
Evaluation

This experiment establishes a codec baseline.

Expected:

Small PSNR drop.
No parameter savings.
No FLOP savings.

---

Required Function

apply_posthoc_420_to_rgb(rgb)

Pipeline:

rgb
→ rgb_to_ycbcr_bt709
→ split Y,Cb,Cr
→ downsample Cb/Cr to H/2 × W/2
→ upsample Cb/Cr to H × W
→ reconstruct YCbCr
→ ycbcr_to_rgb_bt709
→ evaluate

---

# Experiment B
# Neural 4:2:0 Shared Trunk

Purpose:

Train NeRV to predict low-resolution chroma directly.

Question:

Does neural 4:2:0 outperform post-hoc 4:2:0?

---

Architecture

Current NeRV:

Input
 ↓
MLP Stem
 ↓
Upsampling Blocks
 ↓
Final Conv
 ↓
3-channel Output

New Shared-Trunk Version:

Input
 ↓
MLP Stem
 ↓
Shared Full-Resolution Trunk
 ↓
 ├── Y Head
 │      H × W
 │
 └── CbCr Head
        H/2 × W/2

---

Outputs

Y:
[B,1,H,W]

CbCr:
[B,2,H/2,W/2]

RGB Reconstruction:
[B,3,H,W]

---

Training Targets

RGB
 ↓
YCbCr

Y Target:
H × W

CbCr Target:
Downsampled to H/2 × W/2

---

Loss

loss_y =
MSE(Y_pred,Y_gt)

loss_c =
MSE(CbCr_pred,CbCr_gt_low)

Total:

loss =
lambda_y * loss_y +
lambda_c * loss_c

Optional:

rgb_pred =
YCbCr420_to_RGB()

loss_rgb =
MSE(rgb_pred,rgb_gt)

loss += lambda_rgb * loss_rgb

---

Evaluation

Y_pred
CbCr_pred_low

↓

Upsample chroma

↓

YCbCr

↓

RGB

↓

Compute:

RGB PSNR
RGB MS-SSIM
PSNR-Y
PSNR-Cb
PSNR-Cr
FPS

---

# Experiment C
# Neural 4:2:0 Split Branch Architecture

Purpose:

Reduce high-resolution chroma computation.

Question:

Can chroma avoid the final expensive upsampling stage?

---

NeRV-S Spatial Progression

9×16
↓
45×80
↓
90×160
↓
180×320
↓
360×640
↓
720×1280

---

Split Point

Split after:

360×640

---

Architecture

Input
 ↓
MLP Stem
 ↓
Shared Layers
 ↓
360×640 Feature Map
 │
 ├──────────────► Chroma Branch
 │                 CbCr Output
 │                 360×640
 │
 └──────────────► Y Branch
                   Final Upsample
                   720×1280
                   Y Output

---

Outputs

Y:
720×1280

Cb:
360×640

Cr:
360×640

---

Expected Benefits

Compared to Shared Version:

- Lower FLOPs
- Higher FPS
- Similar RGB Quality
- Similar PSNR-Y

---

# Required Metrics

For every experiment:

RGB PSNR
RGB MS-SSIM
PSNR-Y
PSNR-Cb
PSNR-Cr

Optional:

LPIPS

---

Efficiency Metrics

Parameters
Checkpoint Size
Model FPS
End-to-End FPS

Optional:

FLOPs

---

# CSV Output

Create:

results/chroma_nerv_results.csv

Append one row per evaluation.

Columns:

timestamp
run_name
experiment
dataset
epochs
checkpoint
color_space
params_M
checkpoint_size_MB
model_fps
end_to_end_fps
rgb_psnr
rgb_ms_ssim
psnr_y
psnr_cb
psnr_cr
output_sample_ratio
visual_dir
out_dir
notes

---

# Visual Outputs

Save:

pred_rgb_0000.png
gt_rgb_0000.png
error_rgb_0000.png

pred_y_0000.png
gt_y_0000.png

pred_cb_0000.png
gt_cb_0000.png

pred_cr_0000.png
gt_cr_0000.png

For 4:2:0:

pred_cb_low_0000.png
pred_cr_low_0000.png

pred_cb_full_0000.png
pred_cr_full_0000.png

---

# CLI

New File:

train_chroma_nerv.py

Add:

--experiment

Choices:

rgb444
ycbcr444
posthoc420
neural420_shared
neural420_split

Additional:

--lambda_y
--lambda_c
--lambda_rgb

--chroma_downsample

Choices:
area
bilinear
bicubic

--chroma_upsample

Choices:
nearest
bilinear
bicubic

--results_csv
--run_name

---

# Success Criteria

Experiment A:
Post-hoc 4:2:0 works on existing RGB and YCbCr checkpoints.

Experiment B:
Neural 4:2:0 shared trunk trains and evaluates successfully.

Experiment C:
Split architecture trains and evaluates successfully.

All experiments:

- save eval.txt
- save CSV row
- save visual outputs
- compute all metrics

---

# Final Desired Comparison Table

| Experiment | Params | FPS | RGB PSNR | PSNR-Y | PSNR-Cb | PSNR-Cr | MS-SSIM |
|------------|--------|-----|----------|---------|----------|----------|----------|
| RGB 4:4:4 | ? | ? | ? | ? | ? | ? | ? |
| YCbCr 4:4:4 | ? | ? | ? | ? | ? | ? | ? |
| PostHoc 4:2:0 | ? | ? | ? | ? | ? | ? | ? |
| Neural 4:2:0 Shared | ? | ? | ? | ? | ? | ? | ? |
| Neural 4:2:0 Split | ? | ? | ? | ? | ? | ? | ? |

Primary Scientific Comparison:

Neural 4:2:0 Shared
vs
Post-Hoc 4:2:0

Secondary Scientific Comparison:

Neural 4:2:0 Split
vs
Neural 4:2:0 Shared

to determine whether chroma-aware architecture yields efficiency gains while preserving quality.
