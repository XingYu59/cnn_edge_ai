"""
Memory 分析 (Phase 5): 代表模型 + Params/FM/Flatten 与 Memory 关系
================================================================
对已有代表性 CNN 生成统一 Memory Profile, 分析:
  - Params → Weight Memory
  - Feature Map → Activation Memory
  - Depth / Channel / Flatten → Memory
重点验证: "Params 是否足以代表 CNN Memory?"

输出: results/memory_profiles.csv
图:   docs/figures/params_vs_memory.png, activation_vs_memory.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from hpm.architecture import analyze_architecture
from hpm.memory import memory_profile

FIG_DIR = os.path.join(BASE_DIR, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120

# 代表性模型 (小/大/大FM/大channel/深层/大flatten)
REPRESENTATIVES = [
    ('d3_k3', dict(num_classes=43, input_size=64, depth=3,
                   channels=[16, 32, 32], kernel_size=3)),
    ('d5_k3', dict(num_classes=43, input_size=64, depth=5,
                   channels=[32, 32, 64, 64, 128], kernel_size=3)),
    ('d5_k5', dict(num_classes=43, input_size=64, depth=5,
                   channels=[32, 32, 64, 64, 128], kernel_size=5)),
    ('cnn_test', dict(num_classes=43, input_size=64, depth=4,
                      channels=[16, 32, 64, 64], kernel_size=3)),
    ('C1_bigFM', dict(num_classes=43, input_size=64, depth=4,
                      channels=[32, 32, 64, 64], kernel_size=3,
                      pool_positions=[4])),
    ('C3_smallFM', dict(num_classes=43, input_size=64, depth=4,
                        channels=[32, 32, 64, 64], kernel_size=3,
                        pool_positions=[2, 3, 4])),
    ('D3_bigFlatten', dict(num_classes=43, input_size=64, depth=4,
                           channels=[32, 64, 64, 128], kernel_size=3,
                           pool_positions=[2, 4])),
    ('A5_deep', dict(num_classes=43, input_size=64, depth=7,
                     channels=[16, 32, 32, 64, 64, 64, 64], kernel_size=3,
                     pool_positions=[2, 7])),
    ('B2_wideK', dict(num_classes=43, input_size=64, depth=4,
                      channels=[32, 32, 64, 64], kernel_size=[5, 5, 5, 5],
                      pool_positions=[2, 4])),
    ('V11_bigFM', dict(num_classes=43, input_size=64, depth=3,
                       channels=[32, 32, 64], kernel_size=3,
                       pool_positions=[3])),
]


def main():
    rows = []
    for mid, cfg in REPRESENTATIVES:
        p = analyze_architecture(cfg)
        for prec in ('int8', 'fp16'):
            m = memory_profile(p, weight_precision=prec,
                               activation_precision=prec)
            rows.append({
                'model_id': mid,
                'precision': prec,
                'params': m['params'],
                'weight_memory_bytes': m['weight_memory_bytes'],
                'peak_activation_memory_bytes':
                    m['peak_activation_memory_bytes'],
                'estimated_peak_memory_bytes':
                    m['estimated_peak_memory_bytes'],
                'max_feature_map': m['max_feature_map'],
                'max_feature_map_shape': str(m['max_feature_map_shape']),
                'flatten_dim': p['flatten_dim'],
                'depth': p['depth'],
                'total_macs': p['total_macs'],
                'largest_layer': m['largest_activation_layer'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, 'results', 'memory_profiles.csv'),
              index=False)

    # ---- 表格 (int8) ----
    d8 = df[df['precision'] == 'int8']
    print('===== Memory Profile (INT8) =====')
    print(d8[['model_id', 'params', 'weight_memory_bytes',
              'peak_activation_memory_bytes',
              'estimated_peak_memory_bytes', 'max_feature_map_shape',
              'flatten_dim']].to_string(index=False))

    # ---- 关系分析 ----
    print('\n===== Params vs Weight Memory (完全线性, 定义如此) =====')
    w = d8['weight_memory_bytes'].values
    par = d8['params'].values
    print(f'  Pearson(params, weight)=1.0 (weight=params×1, 定义恒等)')

    print('\n===== Params vs Estimated Peak Memory (非线性, 关键) =====')
    est = d8['estimated_peak_memory_bytes'].values
    corr = np.corrcoef(par, est)[0, 1]
    print(f'  Pearson(params, estimated_peak)={corr:.3f}')
    print('  → 若 < 0.95, 说明 activation 部分显著影响总内存')

    # d3_k3 vs A5 对比 (Params 类似但 FM 不同)
    print('\n===== 关键对比: Params 相近但 Memory 不同 =====')
    d3 = d8[d8['model_id'] == 'd3_k3'].iloc[0]
    a5 = d8[d8['model_id'] == 'A5_deep'].iloc[0]
    print(f'  d3_k3 : params={d3["params"]:,} weight={d3["weight_memory_bytes"]:,}B '
          f'peak_act={d3["peak_activation_memory_bytes"]:,}B '
          f'est={d3["estimated_peak_memory_bytes"]:,}B')
    print(f'  A5_deep: params={a5["params"]:,} weight={a5["weight_memory_bytes"]:,}B '
          f'peak_act={a5["peak_activation_memory_bytes"]:,}B '
          f'est={a5["estimated_peak_memory_bytes"]:,}B')
    if d3['params'] > a5['params'] * 0.8:
        print('  → 两者 Params 接近, 但 Memory 不同 → Params 不能代表 Memory!')

    # ---- 图 ----
    print('\n===== 图 =====')
    # Fig: Params vs Memory (weight / estimated)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = par / 1e6
    axes[0].scatter(x, w / 1e6, s=40, c='steelblue')
    axes[0].set_xlabel('Params (M)')
    axes[0].set_ylabel('Weight Memory (MB)')
    axes[0].set_title('Params vs Weight Memory (线性恒等)')
    axes[0].grid(alpha=0.3)
    axes[1].scatter(x, est / 1e6, s=40, c='coral')
    for _, r in d8.iterrows():
        axes[1].annotate(r['model_id'], (r['params'] / 1e6,
                                         r['estimated_peak_memory_bytes'] / 1e6),
                         fontsize=6, textcoords='offset points', xytext=(3, 3))
    axes[1].set_xlabel('Params (M)')
    axes[1].set_ylabel('Estimated Peak Memory (MB)')
    axes[1].set_title(f'Params vs Estimated Peak Memory (corr={corr:.3f})')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'params_vs_memory.png'))
    plt.close(fig)
    print('  params_vs_memory.png')

    # Fig: Activation Memory 占比 (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mids = d8['model_id'].tolist()
    wm = d8['weight_memory_bytes'].values / 1e6
    am = d8['peak_activation_memory_bytes'].values / 1e6
    ax.bar(mids, wm, label='weight (MB)', color='steelblue')
    ax.bar(mids, am, bottom=wm, label='peak activation (MB)', color='coral')
    for i, (w_, a_) in enumerate(zip(wm, am)):
        ax.text(i, w_ + a_ + 0.02, f'act {a_:.2f}MB', ha='center', fontsize=6)
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Weight vs Peak Activation Memory (INT8)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'activation_vs_memory.png'))
    plt.close(fig)
    print('  activation_vs_memory.png')

    print(f'\n===== 完成: results/memory_profiles.csv ({len(df)} 行) =====')


if __name__ == '__main__':
    main()
