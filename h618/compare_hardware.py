"""
双平台横向对比 (Phase 11)
=========================
RK3566 NPU (rk3566/results) vs H618 CPU/Vulkan (h618/results)
相同模型直接比较硬件差异。
输出表 + docs/figures/rk3566_vs_h618_latency.png
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RK = os.path.join(CNN_DIR, 'rk3566', 'results')
H6 = os.path.join(CNN_DIR, 'h618', 'results')
FIG_DIR = os.path.join(CNN_DIR, 'h618', 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


def main():
    # RK3566 NPU latency (静态特征从 h618_dataset 合并, 同结构)
    b = pd.read_csv(os.path.join(RK, 'benchmark_results.csv'))
    v = pd.read_csv(os.path.join(RK, 'validation_benchmark.csv'))
    rk = pd.concat([b[['model_id', 'npu_latency_us']],
                    v[['model_id', 'npu_latency_us']]], ignore_index=True)
    rk = rk.drop_duplicates('model_id')

    # H618
    h = pd.read_csv(os.path.join(H6, 'h618_dataset.csv'))

    df = h.merge(rk[['model_id', 'npu_latency_us']], on='model_id',
                 how='inner')
    df['rk3566_ms'] = df['npu_latency_us'] / 1000
    df['h618cpu_over_rk'] = df['cpu_mean_ms'] / df['rk3566_ms']
    df['h618vk_over_rk'] = df['vulkan_mean_ms'] / df['rk3566_ms']

    print('===== RK3566 NPU vs H618 (相同模型) =====')
    print(df[['model_id', 'total_macs', 'rk3566_ms', 'cpu_mean_ms',
              'vulkan_mean_ms', 'h618cpu_over_rk',
              'h618vk_over_rk']].to_string(index=False))
    print(f'\nH618 CPU / RK3566 平均: {df["h618cpu_over_rk"].mean():.1f}x '
          f'(范围 {df["h618cpu_over_rk"].min():.1f}-{df["h618cpu_over_rk"].max():.1f}x)')
    print(f'H618 Vulkan / RK3566 平均: {df["h618vk_over_rk"].mean():.1f}x')

    # 图: 双平台延迟对比 (分组条形或散点)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 散点: RK3566 vs H618 CPU
    axes[0].scatter(df['rk3566_ms'], df['cpu_mean_ms'], s=50, c='steelblue',
                    zorder=3, label='H618 CPU')
    axes[0].scatter(df['rk3566_ms'], df['vulkan_mean_ms'], s=50, c='coral',
                    zorder=3, marker='^', label='H618 Vulkan')
    lim = [0, max(df['cpu_mean_ms'].max(), df['vulkan_mean_ms'].max()) * 1.05]
    axes[0].plot(lim, lim, 'k--', lw=1, label='y=x')
    axes[0].set_xlabel('RK3566 NPU latency (ms)')
    axes[0].set_ylabel('H618 latency (ms)')
    axes[0].set_title('RK3566 vs H618 (同模型)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # ratio 条形
    d2 = df.sort_values('total_macs')
    xpos = range(len(d2))
    axes[1].bar([i - 0.2 for i in xpos], d2['h618cpu_over_rk'], width=0.4,
                label='H618 CPU / RK3566', color='steelblue', alpha=0.85)
    axes[1].bar([i + 0.2 for i in xpos], d2['h618vk_over_rk'], width=0.4,
                label='H618 Vulkan / RK3566', color='coral', alpha=0.85)
    axes[1].set_xticks(list(xpos))
    axes[1].set_xticklabels(d2['model_id'], rotation=45, fontsize=7)
    axes[1].set_ylabel('latency ratio (×RK3566)')
    axes[1].set_title('延迟倍数 (RK3566=1)')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'rk3566_vs_h618_latency.png'))
    plt.close(fig)
    print('\n图: rk3566_vs_h618_latency.png')
    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
