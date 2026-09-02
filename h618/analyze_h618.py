"""
H618 Latency 分析 (Phase 7-9)
=============================
读 h618_dataset.csv, 分析:
  - MACs/Params/Conv/Linear/Flatten vs CPU/Vulkan latency 相关性
  - CPU vs Vulkan 对比 (ratio / crossover 观察)
图输出 docs/figures/ (cnn/h618 下)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H618_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(H618_DIR, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan


def scatter_fit(ax, x, y, labels, xlabel, title, xunit='M'):
    xv = x / 1e6 if xunit == 'M' else x
    ax.scatter(xv, y, s=40, c='steelblue', zorder=3)
    A = np.vstack([xv, np.ones(len(xv))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, coef[0] * xs + coef[1], 'r--', lw=1, label=f'R²={r2:.3f}')
    for i, l in enumerate(labels):
        ax.annotate(l, (xv[i], y[i]), fontsize=6,
                    textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('latency (ms)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    df = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'))

    # ---- 相关性 ----
    print('===== 相关性 (vs latency) =====')
    metrics = [('total_macs', 'MACs'), ('params', 'Params'),
               ('conv_macs', 'Conv MACs'), ('linear_macs', 'Linear MACs'),
               ('flatten_dim', 'Flatten Dim')]
    print(f'{"指标":<12}{"CPU Pearson":>12}{"Vulkan Pearson":>15}')
    for col, name in metrics:
        cp = pearson(df[col], df['cpu_mean_ms'])
        vp = pearson(df[col], df['vulkan_mean_ms'])
        print(f'{name:<12}{cp:>12.3f}{vp:>15.3f}')

    # ---- CPU vs Vulkan ratio 排序 ----
    print('\n===== CPU/Vulkan ratio (CPU=1, 越小 Vulkan 越慢) =====')
    s = df.sort_values('cpu_vulkan_ratio', ascending=False)
    print(s[['model_id', 'cpu_mean_ms', 'vulkan_mean_ms',
             'cpu_vulkan_ratio']].to_string(index=False))
    print('\nVulkan 相对最好的模型 (ratio>0.6):',
          s[s['cpu_vulkan_ratio'] > 0.6]['model_id'].tolist())

    # ---- 受控观察 ----
    print('\n===== 受控趋势 (A depth / B kernel / C FM) =====')
    a = df[df['model_id'].isin(['A1', 'A5'])]
    print('Depth A1(3层) vs A5(7层): CPU',
          f'{a[a.model_id=="A1"]["cpu_mean_ms"].values[0]:.1f}→'
          f'{a[a.model_id=="A5"]["cpu_mean_ms"].values[0]:.1f}ms')
    c = df[df['model_id'].isin(['C1', 'C3'])]
    print('FM C1(65536) vs C3(4096): CPU',
          f'{c[c.model_id=="C1"]["cpu_mean_ms"].values[0]:.1f}→'
          f'{c[c.model_id=="C3"]["cpu_mean_ms"].values[0]:.1f}ms')

    # ---- 图 ----
    print('\n===== 图 =====')
    labels = df['model_id'].tolist()
    # 2x2: CPU MACs / Vulkan MACs / CPU vs Vulkan
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    scatter_fit(axes[0, 0], df['total_macs'].values,
                df['cpu_mean_ms'].values, labels,
                'Total MACs (M)', 'H618 CPU: MACs vs Latency')
    scatter_fit(axes[0, 1], df['total_macs'].values,
                df['vulkan_mean_ms'].values, labels,
                'Total MACs (M)', 'H618 Vulkan: MACs vs Latency')
    # CPU vs Vulkan 散点
    axes[1, 0].scatter(df['cpu_mean_ms'], df['vulkan_mean_ms'], s=45,
                       c='coral', zorder=3)
    lim = [0, max(df['cpu_mean_ms'].max(), df['vulkan_mean_ms'].max()) * 1.05]
    axes[1, 0].plot(lim, lim, 'k--', lw=1, label='y=x (CPU=Vulkan)')
    for _, r in df.iterrows():
        axes[1, 0].annotate(r['model_id'],
                            (r['cpu_mean_ms'], r['vulkan_mean_ms']),
                            fontsize=6, textcoords='offset points',
                            xytext=(3, 3))
    axes[1, 0].set_xlabel('CPU latency (ms)')
    axes[1, 0].set_ylabel('Vulkan latency (ms)')
    axes[1, 0].set_title('H618: CPU vs Vulkan (全部 Vulkan 更慢)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    # ratio vs MACs
    axes[1, 1].scatter(df['total_macs'] / 1e6, df['cpu_vulkan_ratio'],
                       s=45, c='steelblue', zorder=3)
    for _, r in df.iterrows():
        axes[1, 1].annotate(r['model_id'],
                            (r['total_macs'] / 1e6, r['cpu_vulkan_ratio']),
                            fontsize=6, textcoords='offset points',
                            xytext=(3, 3))
    axes[1, 1].axhline(1, color='k', ls='--', lw=1)
    axes[1, 1].set_xlabel('Total MACs (M)')
    axes[1, 1].set_ylabel('CPU/Vulkan ratio')
    axes[1, 1].set_title('CPU/Vulkan ratio vs MACs (越高 Vulkan 越接近 CPU)')
    axes[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'h618_cpu_vulkan_analysis.png'))
    plt.close(fig)
    print('  h618_cpu_vulkan_analysis.png')

    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
