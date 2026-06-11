"""Colab-friendly helpers for the ChromaNeRV Bunny ablation v2 runs.

Import this file from a notebook cell, choose the sweep values, and pass a
selected command list to run_commands(). Commands use the active Python
interpreter so they work in Colab without shell variable expansion.
"""

import subprocess
import sys


RESULTS_CSV = 'results/chroma420_bunny_ablation_v2.csv'
OUT_DIR = 'output/chroma420_ablation_v2'


def common_args():
    return [
        sys.executable,
        'train_chroma_nerv.py',
        '-e', '300',
        '--lower-width', '96',
        '--num-blocks', '1',
        '--dataset', 'bunny',
        '--frame_gap', '1',
        '--outf', OUT_DIR,
        '--embed', '1.25_40',
        '--stem_dim_num', '512_1',
        '--reduction', '2',
        '--fc_hw_dim', '9_16_26',
        '--expansion', '1',
        '--single_res',
        '--loss_type', 'L2',
        '--warmup', '0.2',
        '--lr_type', 'cosine',
        '--strides', '5', '2', '2', '2', '2',
        '--conv_type', 'conv',
        '-b', '1',
        '--lr', '0.0005',
        '--norm', 'none',
        '--act', 'swish',
        '--results_csv', RESULTS_CSV,
        '--visual_frames', '3',
        '--fps_repeats', '30',
        '--overwrite',
    ]


def loss_sweep(lambda_c_values=(1.5, 2.0, 3.0), lambda_rgb_values=(0.0,)):
    commands = []
    for lambda_c in lambda_c_values:
        for lambda_rgb in lambda_rgb_values:
            commands.append(common_args() + [
                '--experiment', 'neural420_shared',
                '--lambda_y', '1.0',
                '--lambda_c', str(lambda_c),
                '--lambda_rgb', str(lambda_rgb),
                '--chroma_downsample', 'area',
                '--chroma_upsample', 'bilinear',
                '--chroma_scale', '2',
                '--ablation_group', 'loss_sweep',
                '--run_name', f'neural420_shared_lc{lambda_c}_lrgb{lambda_rgb}',
            ])
    return commands


def learned_upsampler_sweep(best_lambda_c, best_lambda_rgb, widths=(8, 16, 32)):
    commands = []
    for width in widths:
        commands.append(common_args() + [
            '--experiment', 'neural420_shared_learned_up',
            '--lambda_y', '1.0',
            '--lambda_c', str(best_lambda_c),
            '--lambda_rgb', str(best_lambda_rgb),
            '--chroma_downsample', 'area',
            '--chroma_upsampler', 'learned',
            '--learned_upsampler_width', str(width),
            '--learned_upsampler_depth', '2',
            '--learned_upsampler_residual',
            '--chroma_scale', '2',
            '--ablation_group', 'learned_upsampler',
            '--run_name', f'neural420_learnedup_w{width}',
        ])
    return commands


def early_chroma_runs(best_lambda_c, best_lambda_rgb, upsamplers=('bilinear',)):
    commands = []
    for upsampler in upsamplers:
        commands.append(common_args() + [
            '--experiment', 'neural420_early_chroma',
            '--lambda_y', '1.0',
            '--lambda_c', str(best_lambda_c),
            '--lambda_rgb', str(best_lambda_rgb),
            '--chroma_downsample', 'area',
            '--chroma_upsampler', upsampler,
            '--chroma_scale', '4',
            '--ablation_group', 'early_chroma',
            '--run_name', f'neural420_chromascale4_{upsampler}',
        ])
    return commands


def y_width_sweep(best_lambda_c, best_lambda_rgb, widths=(96, 80, 64, 48)):
    commands = []
    for width in widths:
        commands.append(common_args() + [
            '--experiment', 'neural420_asym_y',
            '--lambda_y', '1.0',
            '--lambda_c', str(best_lambda_c),
            '--lambda_rgb', str(best_lambda_rgb),
            '--chroma_downsample', 'area',
            '--chroma_upsample', 'bilinear',
            '--chroma_scale', '2',
            '--y_branch_width', str(width),
            '--ablation_group', 'y_width_sweep',
            '--run_name', f'neural420_asymy_w{width}',
        ])
    return commands


def run_commands(commands):
    for command in commands:
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True)


def summarize(csv_path=RESULTS_CSV, baseline_run='posthoc420_from_rgb_nervs'):
    import pandas as pd

    columns = [
        'ablation_group',
        'run_name',
        'experiment',
        'lambda_c',
        'lambda_rgb',
        'chroma_scale',
        'chroma_upsampler',
        'learned_upsampler_width',
        'y_branch_width',
        'params_M',
        'estimated_gflops',
        'model_fps',
        'end_to_end_fps',
        'rgb_psnr',
        'rgb_ms_ssim',
        'psnr_y',
        'psnr_cb',
        'psnr_cr',
    ]
    dataframe = pd.read_csv(csv_path)
    summary = dataframe[columns].copy()
    baseline_rows = dataframe[dataframe['run_name'] == baseline_run]
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[-1]
        for metric in ['rgb_psnr', 'rgb_ms_ssim', 'psnr_y', 'psnr_cb', 'psnr_cr']:
            summary[f'delta_{metric}'] = dataframe[metric] - baseline[metric]
    summary.to_csv('results/chroma420_bunny_ablation_v2_summary.csv', index=False)
    return summary
