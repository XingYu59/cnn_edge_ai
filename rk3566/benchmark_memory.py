"""
RK3566 Memory Benchmark (Phase 7/8)
===================================
用 RKNN eval_memory (init_runtime eval_mem=True) 实测代表模型的
Runtime Memory, 与静态估算对比:

  estimated weight vs measured weight
  estimated peak (weight+act) vs measured total

输出: results/memory_benchmark.csv
图:   docs/figures/estimated_vs_measured_memory.png
      docs/figures/memory_prediction_error.png
"""
import os
import re
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

# 代表模型 (model_id -> rknn 文件) —— 全部已有转换产物
REPRESENTATIVES = [
    ('d3_k3', 'models/cnn_d3_k3.rknn',
     dict(num_classes=43, input_size=64, depth=3, channels=[16, 32, 32],
          kernel_size=3)),
    ('d5_k3', 'models/cnn_d5_k3.rknn',
     dict(num_classes=43, input_size=64, depth=5,
          channels=[32, 32, 64, 64, 128], kernel_size=3)),
    ('d5_k5', 'models/cnn_d5_k5.rknn',
     dict(num_classes=43, input_size=64, depth=5,
          channels=[32, 32, 64, 64, 128], kernel_size=5)),
    ('cnn_test', 'models/cnn_test.rknn',
     dict(num_classes=43, input_size=64, depth=4, channels=[16, 32, 64, 64],
          kernel_size=3)),
    ('C1_bigFM', 'models/exp_C1.rknn',
     dict(num_classes=43, input_size=64, depth=4, channels=[32, 32, 64, 64],
          kernel_size=3, pool_positions=[4])),
    ('C3_smallFM', 'models/exp_C3.rknn',
     dict(num_classes=43, input_size=64, depth=4, channels=[32, 32, 64, 64],
          kernel_size=3, pool_positions=[2, 3, 4])),
    ('D3_bigFlatten', 'models/exp_D3.rknn',
     dict(num_classes=43, input_size=64, depth=4,
          channels=[32, 64, 64, 128], kernel_size=3, pool_positions=[2, 4])),
    ('A5_deep', 'models/exp_A5.rknn',
     dict(num_classes=43, input_size=64, depth=7,
          channels=[16, 32, 32, 64, 64, 64, 64], kernel_size=3,
          pool_positions=[2, 7])),
    ('B2_wideK', 'models/exp_B2.rknn',
     dict(num_classes=43, input_size=64, depth=4, channels=[32, 32, 64, 64],
          kernel_size=[5, 5, 5, 5], pool_positions=[2, 4])),
    ('V11_bigFM', 'models/val_V11.rknn',
     dict(num_classes=43, input_size=64, depth=3, channels=[32, 32, 64],
          kernel_size=3, pool_positions=[3])),
]


def measure_memory(rknn_path: str) -> dict:
    """实测 RKNN Runtime Memory (eval_memory)。"""
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0, f'load failed: {rknn_path}'
    rknn.init_runtime(target='rk3566', eval_mem=True)
    try:
        mem = rknn.eval_memory()
    finally:
        rknn.release()
    # eval_memory 返回 dict (is_print=False 时) 或 None
    if isinstance(mem, dict):
        return {
            'measured_weight_bytes': mem['weight_memory'],
            'measured_internal_bytes': mem['internal_memory'],
            'measured_total_bytes': mem['total_memory'],
        }
    # 解析打印文本 (兜底)
    text = str(mem)
    m = re.search(r'Weight Memory:\s*([\d.]+)\s*KiB', text)
    return {'measured_total_bytes': None}


def main():
    rows = []
    for mid, rknn_path, cfg in REPRESENTATIVES:
        print(f'===== {mid} =====')
        p = analyze_architecture(cfg)
        est = memory_profile(p, 'int8', 'int8')   # 静态估算
        meas = measure_memory(rknn_path)

        rows.append({
            'model_id': mid,
            'params': p['params'],
            'est_weight_bytes': est['weight_memory_bytes'],
            'est_peak_activation_bytes': est['peak_activation_memory_bytes'],
            'est_total_bytes': est['estimated_peak_memory_bytes'],
            'meas_weight_bytes': meas.get('measured_weight_bytes'),
            'meas_internal_bytes': meas.get('measured_internal_bytes'),
            'meas_total_bytes': meas.get('measured_total_bytes'),
        })
        print(f'  est: weight={est["weight_memory_bytes"]:,}B '
              f'act={est["peak_activation_memory_bytes"]:,}B '
              f'total={est["estimated_peak_memory_bytes"]:,}B')
        print(f'  meas: weight={meas.get("measured_weight_bytes")}B '
              f'internal={meas.get("measured_internal_bytes")}B '
              f'total={meas.get("measured_total_bytes")}B')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, 'results', 'memory_benchmark.csv'),
              index=False)

    # ---- 误差分析 ----
    print('\n===== 误差分析 =====')
    valid = df[df['meas_total_bytes'].notna()]
    w_err = ((valid['meas_weight_bytes'] - valid['est_weight_bytes'])
             / valid['meas_weight_bytes'] * 100)
    t_err = ((valid['meas_total_bytes'] - valid['est_total_bytes'])
             / valid['meas_total_bytes'] * 100)
    print('  Weight Memory: 平均相对误差 '
          f'{w_err.abs().mean():.1f}% (max {w_err.abs().max():.1f}%)')
    print('  Total Memory : 平均相对误差 '
          f'{t_err.abs().mean():.1f}% (max {t_err.abs().max():.1f}%)')
    print(f'  Total MAE: {(valid["meas_total_bytes"] - valid["est_total_bytes"]).abs().mean():,.0f}B')
    # 相关性
    c1 = np.corrcoef(valid['est_weight_bytes'], valid['meas_weight_bytes'])[0, 1]
    c2 = np.corrcoef(valid['est_total_bytes'], valid['meas_total_bytes'])[0, 1]
    print(f'  Pearson(est_weight, meas_weight)={c1:.4f}')
    print(f'  Pearson(est_total, meas_total)={c2:.4f}')

    # ---- 图 ----
    print('\n===== 图 =====')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # estimated vs measured total
    axes[0].scatter(valid['est_total_bytes'] / 1e6,
                    valid['meas_total_bytes'] / 1e6, s=50, c='steelblue',
                    zorder=3)
    lim = [min(valid['est_total_bytes'].min(),
               valid['meas_total_bytes'].min()) * 0.9 / 1e6,
           max(valid['est_total_bytes'].max(),
               valid['meas_total_bytes'].max()) * 1.1 / 1e6]
    axes[0].plot(lim, lim, 'r--', lw=1, label='y=x')
    for _, r in valid.iterrows():
        axes[0].annotate(r['model_id'],
                         (r['est_total_bytes'] / 1e6,
                          r['meas_total_bytes'] / 1e6),
                         fontsize=6, textcoords='offset points',
                         xytext=(3, 3))
    axes[0].set_xlabel('Estimated Peak Memory (MB)')
    axes[0].set_ylabel('Measured Total Memory (MB)')
    axes[0].set_title(f'Estimated vs Measured Memory (corr={c2:.3f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # prediction error
    axes[1].bar(valid['model_id'], t_err, color='coral', alpha=0.8)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_ylabel('Total Memory error (%)')
    axes[1].set_title('Memory Prediction Error by Model')
    axes[1].tick_params(axis='x', rotation=30, labelsize=7)
    axes[1].grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'estimated_vs_measured_memory.png'))
    plt.close(fig)
    print('  estimated_vs_measured_memory.png')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xpos = np.arange(len(valid))
    ax.bar(xpos - 0.2, w_err, width=0.4, label='weight err %',
           color='steelblue', alpha=0.8)
    ax.bar(xpos + 0.2, t_err, width=0.4, label='total err %',
           color='coral', alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(valid['model_id'], rotation=30, fontsize=7)
    ax.set_ylabel('error (%)')
    ax.set_title('Memory Prediction Error (weight vs total)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'memory_prediction_error.png'))
    plt.close(fig)
    print('  memory_prediction_error.png')

    print(f'\n===== 完成: results/memory_benchmark.csv =====')


if __name__ == '__main__':
    main()
