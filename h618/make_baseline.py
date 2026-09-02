"""
Phase 1: Baseline Summary (14 模型双平台数据)
=============================================
合并 RK3566 + H618 已有数据 → h618_baseline_summary.csv
"""
import os
import sys

import pandas as pd

CNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RK = os.path.join(CNN_DIR, 'rk3566', 'results')
H6 = os.path.join(CNN_DIR, 'h618', 'results')


def main():
    h = pd.read_csv(os.path.join(H6, 'h618_dataset.csv'))
    b = pd.read_csv(os.path.join(RK, 'benchmark_results.csv'))
    v = pd.read_csv(os.path.join(RK, 'validation_benchmark.csv'))
    rk = pd.concat([b[['model_id', 'npu_latency_us']],
                    v[['model_id', 'npu_latency_us']]],
                   ignore_index=True).drop_duplicates('model_id')
    rk.columns = ['model_id', 'rk3566_us']

    df = h.merge(rk, on='model_id', how='left')
    df['rk3566_ms'] = df['rk3566_us'] / 1000
    out_cols = ['model_id', 'total_macs', 'conv_macs', 'linear_macs',
                'flatten_dim', 'params', 'depth',
                'rk3566_ms', 'cpu_mean_ms', 'vulkan_mean_ms']
    df[out_cols].to_csv(os.path.join(H6,
                                     'h618_baseline_summary.csv'),
                        index=False)
    print(f'baseline: {len(df)} 模型')
    print(df[['model_id', 'flatten_dim', 'rk3566_ms', 'cpu_mean_ms',
              'vulkan_mean_ms']].to_string(index=False))


if __name__ == '__main__':
    main()
