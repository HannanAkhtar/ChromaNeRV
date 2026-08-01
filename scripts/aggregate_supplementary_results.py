#!/usr/bin/env python
"""Aggregate Bunny and UVG7 sequence metrics with equal eight-video weighting."""

import argparse
import csv
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nerv_generalization import DEFAULT_CONFIGS, UVG_SEQUENCES
from persistence import atomic_write_json, atomic_write_text, read_json


SEQUENCES = ('Bunny',) + UVG_SEQUENCES
METRICS = (
    'rgb_psnr', 'rgb_ms_ssim', 'psnr_y', 'psnr_cb', 'psnr_cr',
    'yuv_psnr_611_mse', 'yuv_ssim_611', 'lpips_alex', 'dists', 'vmaf',
    'temporal_rgb_error_diff', 'params_M', 'estimated_gflops',
    'gflops_per_output_megapixel', 'model_fps', 'end_to_end_fps',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bunny_root', required=True)
    parser.add_argument('--uvg_root', required=True)
    parser.add_argument('--output_root', default='results/supplementary')
    parser.add_argument('--allow_partial', action='store_true')
    return parser.parse_args()


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def collect_rows(bunny_root, uvg_root):
    rows, missing = [], []
    for sequence in SEQUENCES:
        root = Path(bunny_root) if sequence == 'Bunny' else Path(uvg_root)
        for config in DEFAULT_CONFIGS:
            candidates = (
                root / sequence / config / 'eval_metrics.json',
                root / sequence.lower() / config / 'eval_metrics.json',
                root / config / 'eval_metrics.json',
            )
            found = [candidate for candidate in candidates if candidate.is_file()]
            if len(found) > 1:
                raise RuntimeError(
                    f'Duplicate result files for {sequence}/{config}: '
                    + ', '.join(str(path) for path in found))
            path = found[0] if found else None
            if path is None:
                missing.append(f'{sequence}/{config}')
                continue
            row = read_json(path)
            row.update({
                'sequence': sequence,
                'config_name': config,
                'source_path': f'{sequence}/{config}/eval_metrics.json',
            })
            rows.append(row)
    keys = [(row['sequence'], row['config_name']) for row in rows]
    duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
    if duplicates:
        raise RuntimeError(f'Duplicate sequence/configuration rows: {duplicates}')
    return rows, missing


def mean_metric(rows, metric):
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def aggregate_rows(rows):
    output = []
    full = [row for row in rows if row['config_name'] == 'full_rgb']
    full_params = mean_metric(full, 'params_M')
    full_gflops = mean_metric(full, 'estimated_gflops')
    for config in DEFAULT_CONFIGS:
        selected = [row for row in rows if row['config_name'] == config]
        if not selected:
            continue
        result = {'config_name': config, 'sequence_count': len(selected)}
        result.update({metric: mean_metric(selected, metric) for metric in METRICS})
        result['parameter_reduction_percent'] = (
            100 * (1 - result['params_M'] / full_params)
            if result.get('params_M') is not None and full_params else None)
        result['gflop_reduction_percent'] = (
            100 * (1 - result['estimated_gflops'] / full_gflops)
            if result.get('estimated_gflops') is not None and full_gflops else None)
        output.append(result)
    return output


def delta_rows(rows, pairs, per_sequence=False):
    indexed = {(row['sequence'], row['config_name']): row for row in rows}
    output = []
    sequences = SEQUENCES if per_sequence else ('eight_sequence_average',)
    for left, right in pairs:
        for sequence in sequences:
            for metric in METRICS:
                if per_sequence:
                    left_value = indexed.get((sequence, left), {}).get(metric)
                    right_value = indexed.get((sequence, right), {}).get(metric)
                else:
                    left_value = mean_metric(
                        [r for r in rows if r['config_name'] == left], metric)
                    right_value = mean_metric(
                        [r for r in rows if r['config_name'] == right], metric)
                if left_value is None or right_value is None:
                    continue
                output.append({
                    'sequence': sequence, 'comparison': f'{left} - {right}',
                    'metric': metric, 'delta': float(left_value) - float(right_value),
                })
    return output


def latex_table(rows, fields):
    lines = ['\\begin{tabular}{' + 'l' + 'r' * (len(fields) - 1) + '}',
             ' & '.join(field.replace('_', '\\_') for field in fields) + ' \\\\', '\\hline']
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, '')
            values.append(f'{value:.4f}' if isinstance(value, float) else str(value))
        lines.append(' & '.join(values).replace('_', '\\_') + ' \\\\')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    rows, missing = collect_rows(args.bunny_root, args.uvg_root)
    if missing and not args.allow_partial:
        raise RuntimeError(f'Missing {len(missing)} expected jobs: {", ".join(missing)}')
    if not rows:
        raise RuntimeError('No supplementary eval_metrics.json files were found')
    output = Path(args.output_root)
    aggregate = aggregate_rows(rows)
    matched_pairs = (('chroma_w8', 'rgbsplit_w8'), ('chroma_w4', 'rgbsplit_w4'))
    matched = delta_rows(rows, matched_pairs)
    per_sequence = delta_rows(rows, matched_pairs, per_sequence=True)
    color = delta_rows(rows, (('full_ycbcr444', 'full_rgb'),))
    row_fields = sorted({key for row in rows for key in row})
    aggregate_fields = ['config_name', 'sequence_count', *METRICS,
                        'parameter_reduction_percent', 'gflop_reduction_percent']
    delta_fields = ['sequence', 'comparison', 'metric', 'delta']
    write_csv(output / 'sequence_results.csv', rows, row_fields)
    write_csv(output / 'aggregate_results.csv', aggregate, aggregate_fields)
    write_csv(output / 'matched_deltas.csv', matched, delta_fields)
    write_csv(output / 'per_sequence_matched_deltas.csv', per_sequence, delta_fields)
    write_csv(output / 'color_space_control.csv', color, delta_fields)
    additional = [row for row in rows if any(row.get(k) is not None for k in ('lpips_alex', 'dists', 'vmaf'))]
    write_csv(output / 'additional_metrics.csv', additional, row_fields)
    atomic_write_json(output / 'manifest.json', {
        'complete': not missing, 'expected_jobs': 48, 'included_jobs': len(rows),
        'missing_jobs': missing, 'runs': [row['source_path'] for row in rows],
        'aggregation': 'equal weight per sequence across Bunny and UVG7',
    })
    latex = output / 'latex'
    tables = {
        'aggregate_results.tex': (aggregate, aggregate_fields),
        'primary_vs_full_rgb.tex': (aggregate, aggregate_fields),
        'matched_control.tex': (matched, delta_fields),
        'per_sequence_matched.tex': (per_sequence, delta_fields),
        'color_space_control.tex': (color, delta_fields),
        'additional_metrics.tex': (additional, row_fields),
    }
    for name, (table_rows, fields) in tables.items():
        atomic_write_text(latex / name, latex_table(table_rows, fields))
    print(f'Aggregated {len(rows)} sequence/configuration runs; missing={len(missing)}')


if __name__ == '__main__':
    main()
