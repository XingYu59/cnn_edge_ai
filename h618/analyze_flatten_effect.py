"""
Flatten / Linear 效应分析 (Phase 8-9, Hypothesis H1/H2)
=======================================================
问题: RK3566 对 Linear/Flatten 是否有比 H618 CPU 更严重的惩罚?

方法:
1. 双平台 M2 回归系数对比 (linear 每 M MACs 的开销 vs conv)
   - RK3566: 14 模型实测
   - H618: 25 模型实测
2. FD 组 (flatten 分级): H618 实测 + RK3566 predictor 预测 (标注 predicted)
3. Equal-structure: 找 conv 相近 linear 差异的模型对比

输出图 + 结论。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RK = os.path.join(CNN_DIR, 'rk3566', 'results')
H6 = os.path.join(CNN_DIR, 'h618', 'results')
FIG_DIR = os.path.join(H6, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


def mlt_fit(df, ycol):
    X = np.column_stack([df['conv_macs'].values / 1e6,
                         df['linear_macs'].values / 1e6,
                         np.ones(len(df))])
    y = df[ycol].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yp = X @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, r2


def main():
    h = pd.read_csv(os.path.join(H6, 'h618_dataset.csv'))
    # RK3566: benchmark + validation (仅 latency), 静态从 h 合并
    b = pd.read_csv(os.path.join(RK, 'benchmark_results.csv'))
    v = pd.read_csv(os.path.join(RK, 'validation_benchmark.csv'))
    rk = pd.concat([b[['model_id', 'npu_latency_us']],
                    v[['model_id', 'npu_latency_us']]],
                   ignore_index=True).drop_duplicates('model_id')
    h_s = h[['model_id', 'conv_macs', 'linear_macs', 'total_macs']]
    rk = rk.merge(h_s, on='model_id').dropna(subset=['conv_macs'])

    print('===== 1. M2 回归系数对比 (T = a·Conv + b·Linear + c) =====')
    print('  单位: us/M (每 1M MACs 的开销)')
    for name, df_, ycol in [
        ('RK3566 NPU', rk, 'npu_latency_us'),
        ('H618 CPU', h, 'cpu_mean_ms')]:
        coef, r2 = mlt_fit(df_, ycol)
        unit = 'us/M' if name == 'RK3566 NPU' else 'ms/M'
        print(f'  {name:<12} conv={coef[0]:.3f}{unit} '
              f'linear={coef[1]:.3f}{unit} (c={coef[2]:.1f}, R²={r2:.3f})')
        if name == 'RK3566 NPU':
            rk_coef = coef
        else:
            h_coef = coef
    # 统一单位: us
    print('\n  统一为 us/M:')
    print(f'  RK3566: conv={rk_coef[0]:.1f}us/M linear={rk_coef[1]:.0f}us/M '
          f'ratio(linear/conv)={rk_coef[1]/rk_coef[0]:.0f}x')
    print(f'  H618  : conv={h_coef[0]*1000:.1f}us/M linear={h_coef[1]*1000:.0f}us/M '
          f'ratio(linear/conv)={h_coef[1]/h_coef[0]:.0f}x')

    print('\n===== 2. FD 组 (flatten 分级): H618 实测 vs RK3566 预测 =====')
    # RK3566 predictor (rk3566 实测拟合的 M2): T(us) = 3.023*conv + 980.48*lin + 32
    fd = h[h['model_id'].str.startswith('FD')]
    print(f'{"model":<6}{"flatten":>8}{"convM":>7}{"linM":>7}'
          f'{"H618_cpu":>10}{"RK_pred":>10}{"RK/H618":>9}')
    for _, r in fd.iterrows():
        rk_pred = (3.023 * r['conv_macs'] / 1e6
                   + 980.48 * r['linear_macs'] / 1e6 + 32) / 1000  # ms
        ratio = r['cpu_mean_ms'] / rk_pred if rk_pred > 0 else 0
        print(f'{r["model_id"]:<6}{r["flatten_dim"]:>8,}'
              f'{r["conv_macs"]/1e6:>7.0f}{r["linear_macs"]/1e6:>7.2f}'
              f'{r["cpu_mean_ms"]:>10.2f}{rk_pred:>10.2f}{ratio:>9.1f}')

    print('\n  RK/H618 ratio: 小 = 两平台差距小 (H618 linear-heavy 相对不吃亏)')

    print('\n===== 3. Equal-结构对照 (conv 相近, linear 不同) =====')
    pairs = [('C3', 'D3'), ('V3', 'FD32K')]
    for a_id, b_id in pairs:
        ra = h[h['model_id'] == a_id].iloc[0]
        rb = h[h['model_id'] == b_id].iloc[0]
        print(f'  {a_id}: conv={ra["conv_macs"]/1e6:.0f}M '
              f'lin={ra["linear_macs"]/1e6:.2f}M CPU={ra["cpu_mean_ms"]:.1f}ms | '
              f'{b_id}: conv={rb["conv_macs"]/1e6:.0f}M '
              f'lin={rb["linear_macs"]/1e6:.2f}M CPU={rb["cpu_mean_ms"]:.1f}ms')

    # 图: flatten vs latency (双平台)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # H618 CPU: flatten vs cpu
    axes[0].scatter(h['flatten_dim'], h['cpu_mean_ms'], s=30,
                    c='steelblue', alpha=0.7)
    fd = h[h['model_id'].str.startswith('FD')]
    axes[0].scatter(fd['flatten_dim'], fd['cpu_mean_ms'], s=60,
                    c='red', zorder=4, label='FD 组 (flatten 分级)')
    for _, r in fd.iterrows():
        axes[0].annotate(r['model_id'], (r['flatten_dim'],
                                         r['cpu_mean_ms']),
                         fontsize=7, textcoords='offset points',
                         xytext=(4, 4))
    axes[0].set_xlabel('Flatten dimension')
    axes[0].set_ylabel('H618 CPU latency (ms)')
    axes[0].set_title('H618 CPU: Flatten vs Latency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # RK3566 flatten vs latency (已有实测)
    h2 = h.merge(rk[['model_id', 'npu_latency_us']], on='model_id')
    axes[1].scatter(h2['flatten_dim'], h2['npu_latency_us'] / 1000,
                    s=30, c='coral', alpha=0.7)
    for _, r in h2.iterrows():
        axes[1].annotate(r['model_id'], (r['flatten_dim'],
                                         r['npu_latency_us'] / 1000),
                         fontsize=6, textcoords='offset points',
                         xytext=(3, 3))
    axes[1].set_xlabel('Flatten dimension')
    axes[1].set_ylabel('RK3566 latency (ms)')
    axes[1].set_title('RK3566: Flatten vs Latency (实测)')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'h618_flatten_dual_platform.png'))
    plt.close(fig)
    print('\n图: h618_flatten_dual_platform.png')
    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
