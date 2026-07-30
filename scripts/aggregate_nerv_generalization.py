#!/usr/bin/env python
import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nerv_generalization import DEFAULT_CONFIGS, UVG_SEQUENCES  # noqa: E402
from persistence import (  # noqa: E402
    atomic_copy_directory_files,
    read_json,
    validate_persistent_run,
)


METRICS = (
    'rgb_psnr', 'psnr_y', 'yuv_psnr_611_mse', 'vmaf', 'dists',
    'lpips_alex', 'rgb_ms_ssim', 'params_M', 'estimated_gflops',
)
LOWER_IS_BETTER = {'dists', 'lpips_alex', 'fid'}


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate NeRV UVG7 generalization results')
    parser.add_argument('--results_root', default='output/nerv_generalization')
    parser.add_argument('--output_dir', default='results/nerv_generalization')
    parser.add_argument('--persistent_root', default=None)
    parser.add_argument('--allow_partial', action='store_true')
    return parser.parse_args()


def load_result(results_root, persistent_root, sequence, config):
    local_path = Path(results_root) / sequence / config / 'eval_metrics.json'
    if local_path.is_file():
        row = read_json(local_path)
        row['result_source'] = 'local'
        return row
    if persistent_root:
        persistent_run = Path(persistent_root) / 'runs' / sequence / config
        valid, reason = validate_persistent_run(persistent_run)
        if (persistent_run / 'completion.json').exists() and not valid:
            raise RuntimeError(
                f'Invalid persistent result {sequence}/{config}: {reason}')
        if valid:
            row = read_json(persistent_run / 'eval_metrics.json')
            row['result_source'] = 'persistent'
            return row
    return None


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def equal_sequence_average(rows, key):
    values = finite_values(rows, key)
    return sum(values) / len(values) if values else ''


def paired_deltas(rows, left, right):
    indexed = {(row['sequence'], row['config_name']): row for row in rows}
    output = []
    for sequence in sorted({row['sequence'] for row in rows}):
        if (sequence, left) not in indexed or (sequence, right) not in indexed:
            continue
        for metric in METRICS:
            left_value = indexed[sequence, left].get(metric)
            right_value = indexed[sequence, right].get(metric)
            if left_value is None or right_value is None:
                continue
            output.append({
                'sequence': sequence,
                'comparison': f'{left} - {right}',
                'metric': metric,
                'delta': float(left_value) - float(right_value),
                'direction': 'lower_is_better' if metric in LOWER_IS_BETTER else 'higher_is_better',
            })
    for metric in METRICS:
        selected = [
            row for row in output
            if row['metric'] == metric and row['sequence'] != 'UVG_average'
        ]
        if selected:
            output.append({
                'sequence': 'UVG_average',
                'comparison': f'{left} - {right}',
                'metric': metric,
                'delta': sum(row['delta'] for row in selected) / len(selected),
                'direction': (
                    'lower_is_better' if metric in LOWER_IS_BETTER else 'higher_is_better'
                ),
            })
    return output


def main():
    args = parse_args()
    root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    persistent_root = Path(args.persistent_root).resolve() if args.persistent_root else None
    rows = []
    missing = []
    for sequence in UVG_SEQUENCES:
        for config in DEFAULT_CONFIGS:
            row = load_result(root, persistent_root, sequence, config)
            if row is None:
                missing.append(f'{sequence}/{config}')
                continue
            row.update({'sequence': sequence, 'config_name': config})
            rows.append(row)
    if missing and not args.allow_partial:
        raise RuntimeError(
            'Missing expected results: ' + ', '.join(missing) + '. Use --allow_partial for smoke data.')
    if not rows:
        raise RuntimeError(f'No eval_metrics.json files found under {root}')
    write_csv(output_dir / 'per_sequence_summary.csv', rows)

    averages = []
    for config in DEFAULT_CONFIGS:
        selected = [row for row in rows if row['config_name'] == config]
        if not selected:
            continue
        average = {'config_name': config, 'sequence_count': len(selected)}
        for metric in METRICS:
            average[metric] = equal_sequence_average(selected, metric)
        full = next((row for row in averages if row['config_name'] == 'full_rgb'), None)
        if full and average.get('estimated_gflops') != '':
            average['gflop_reduction_percent'] = 100.0 * (
                1.0 - average['estimated_gflops'] / full['estimated_gflops'])
        else:
            average['gflop_reduction_percent'] = 0.0 if config == 'full_rgb' else ''
        averages.append(average)
    write_csv(output_dir / 'uvg7_average_summary.csv', averages)

    matched = (
        paired_deltas(rows, 'chroma_w8', 'rgbsplit_w8')
        + paired_deltas(rows, 'chroma_w4', 'rgbsplit_w4')
    )
    write_csv(output_dir / 'matched_deltas.csv', matched)
    full_deltas = (
        paired_deltas(rows, 'chroma_w8', 'full_rgb')
        + paired_deltas(rows, 'chroma_w4', 'full_rgb')
    )
    write_csv(output_dir / 'full_rgb_deltas.csv', full_deltas)

    pooled_path = root / 'manifests' / 'pooled_fid.json'
    if not pooled_path.exists() and persistent_root:
        pooled_path = persistent_root / 'manifests' / 'pooled_fid.json'
    pooled = json.loads(pooled_path.read_text(encoding='utf-8')) if pooled_path.exists() else []
    write_csv(output_dir / 'pooled_fid.csv', pooled, ['config_name', 'fid', 'pooled_frame_count'])
    write_csv(
        output_dir / 'rate_summary.csv',
        [{'available': False, 'reason': 'Exact metadata-complete rate codec is unavailable.'}],
    )
    columns = ['config_name', *METRICS, 'gflop_reduction_percent']
    with (output_dir / 'nerv_generalization_table.tex').open('w', encoding='utf-8') as handle:
        handle.write('\\begin{tabular}{l' + 'r' * (len(columns) - 1) + '}\n')
        handle.write(' & '.join(columns).replace('_', '\\_') + ' \\\\\n\\hline\n')
        for row in averages:
            values = [row.get(column, '') for column in columns]
            formatted = [
                f'{value:.4f}' if isinstance(value, float) else str(value)
                for value in values
            ]
            handle.write(' & '.join(formatted).replace('_', '\\_') + ' \\\\\n')
        handle.write('\\end{tabular}\n')

    try:
        import matplotlib.pyplot as plt
        for metric, filename in (
                ('yuv_psnr_611_mse', 'yuv_psnr_vs_gflops.png'),
                ('vmaf', 'vmaf_vs_gflops.png')):
            points = [
                row for row in averages
                if row.get(metric) not in {'', None} and row.get('estimated_gflops') not in {'', None}
            ]
            if not points:
                continue
            plt.figure()
            plt.scatter(
                [row['estimated_gflops'] for row in points],
                [row[metric] for row in points],
            )
            for row in points:
                plt.annotate(row['config_name'], (row['estimated_gflops'], row[metric]))
            plt.xlabel('Model-only GFLOPs')
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=180)
            plt.close()
    except ImportError:
        print('matplotlib unavailable; plot generation skipped')
    if persistent_root:
        atomic_copy_directory_files(output_dir, persistent_root / 'results')


if __name__ == '__main__':
    main()
