"""
Controlled Benchmark 分析 (Task Plan 第二阶段, 第 14/15/16/20 节)
=================================================================
读取 results/benchmark_results.csv, 生成:
  - 相关性分析 (Pearson/Spearman): MACs/Params/Conv MACs/Linear MACs/Flatten vs latency
  - 回归 M1: T = a*MACs + b      (仅总 MACs)
  - 回归 M2: T = a*ConvMACs + b*LinearMACs + c (结构感知)
  - 7 张图 -> docs/figures/
  - controlled_benchmark_report.md

注意 (第 15/18 节): 相关性不是因果; 样本有限时不过度解读;
不声称 MACs 无法预测, 只报告"当前实验显示存在偏差"。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 120

COLORS = {'A_depth': '#1f77b4', 'B_kernel': '#ff7f0e',
          'C_featuremap': '#2ca02c', 'D_classifier': '#d62728',
          'legacy': '#7f7f7f'}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    """手写 Spearman: 秩变换后 Pearson (不依赖 scipy)。"""
    def rank(v):
        order = np.argsort(v, kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v))
        return ranks
    return pearson(rank(np.asarray(x, float)),
                   rank(np.asarray(y, float)))


def linfit(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yp = A @ coef
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    return coef, r2, mae, mape


def mlt_fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yp = X @ coef
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    return coef, r2, mae, mape


def scatter(ax, x, y, df, xlabel, title, xunit='M'):
    xv = x / 1e6 if xunit == 'M' else x
    ax.scatter(xv, y, s=35, c='steelblue', alpha=0.75, zorder=3)
    coef, r2, mae, mape = linfit(xv, y)
    a, b = float(coef[0]), float(coef[1])
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, a * xs + b, 'r--', lw=1,
            label=f'fit (R²={r2:.3f})')
    for i, (_, r) in enumerate(df.iterrows()):
        ax.annotate(r['model_id'], (xv[i], y[i]),
                    textcoords='offset points', xytext=(3, 3), fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('latency (us)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return r2, mae, mape


def group_scatter(ax, df, xcol, ycol, xlabel, title):
    for gid, grp in df.groupby('experiment_group'):
        ax.scatter(grp[xcol], grp[ycol], s=40, alpha=0.8,
                   color=COLORS[gid], label=gid, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'results',
                                  'benchmark_results.csv'))
    df = df[df['npu_latency_us'].notna()]
    lat = df['npu_latency_us'].values

    # ---------- 相关性 ----------
    print('===== 相关性分析 (vs NPU latency) =====')
    metrics = {
        'macs': 'MACs',
        'params': 'Params',
        'conv_macs': 'Conv MACs',
        'linear_macs': 'Linear MACs',
        'flatten_dimension': 'Flatten Dim',
    }
    corr_rows = []
    for col, name in metrics.items():
        p = pearson(df[col], lat)
        s = spearman(df[col], lat)
        corr_rows.append((name, p, s))
        print(f'  {name:<14} Pearson={p:+.3f}  Spearman={s:+.3f}')

    # ---------- 回归 M1 / M2 ----------
    print('\n===== 回归 =====')
    coef1, r2_1, mae1, mape1 = linfit(df['macs'].values / 1e6, lat)
    print(f'  M1: T = {coef1[0]:.3f}*MACs(M) {coef1[1]:+.1f} | '
          f'R²={r2_1:.3f} MAE={mae1:.1f}us MAPE={mape1:.1f}%')

    X2 = np.column_stack([df['conv_macs'].values / 1e6,
                          df['linear_macs'].values / 1e6,
                          np.ones(len(df))])
    coef2, r2_2, mae2, mape2 = mlt_fit(X2, lat)
    print(f'  M2: T = {coef2[0]:.3f}*ConvM {coef2[1]:.3f}*LinM {coef2[2]:+.1f} | '
          f'R²={r2_2:.3f} MAE={mae2:.1f}us MAPE={mape2:.1f}%')

    # ---------- 图 ----------
    print('\n===== 生成图 =====')
    # Fig1-5
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    scatter(axes[0, 0], df['macs'].values, lat, df,
            'Total MACs (M)', 'Fig1: Total MACs vs Latency')
    scatter(axes[0, 1], df['conv_macs'].values,
            df['conv_latency_us'].values, df,
            'Conv MACs (M)', 'Fig2: Conv MACs vs Conv Latency')
    scatter(axes[0, 2], df['linear_macs'].values,
            df['gemm_latency_us'].values, df,
            'Linear MACs (M)', 'Fig3: Linear MACs vs GEMM Latency')
    scatter(axes[1, 0], df['flatten_dimension'].values, lat, df,
            'Flatten Dim', 'Fig4: Flatten Dim vs Latency', xunit='')
    scatter(axes[1, 1], df['flatten_dimension'].values,
            df['gemm_latency_us'].values, df,
            'Flatten Dim', 'Fig5: Flatten Dim vs GEMM Latency', xunit='')
    # Fig6: Depth vs Latency (A 组)
    a_grp = df[df['experiment_group'] == 'A_depth']
    if len(a_grp):
        axes[1, 2].plot(a_grp['depth'], a_grp['npu_latency_us'], 'o-',
                        color=COLORS['A_depth'])
        for _, r in a_grp.iterrows():
            axes[1, 2].annotate(r['model_id'], (r['depth'], r['npu_latency_us']),
                                textcoords='offset points', xytext=(4, 4),
                                fontsize=7)
        axes[1, 2].set_xlabel('depth')
        axes[1, 2].set_ylabel('NPU latency (us)')
        axes[1, 2].set_title('Fig6: Depth vs Latency (A group)')
        axes[1, 2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_controlled_1to6.png'))
    plt.close(fig)
    print('  fig_controlled_1to6.png')

    # Fig7: Kernel vs Latency (B 组)
    b_grp = df[df['experiment_group'] == 'B_kernel']
    if len(b_grp):
        fig, ax = plt.subplots(figsize=(7, 5))
        xs = np.arange(len(b_grp))
        ax.bar(xs, b_grp['npu_latency_us'],
               color=[COLORS['B_kernel']] * len(b_grp), alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{r["model_id"]}\n{r["independent_var"]}'
                            for _, r in b_grp.iterrows()], fontsize=7)
        ax.set_ylabel('NPU latency (us)')
        ax.set_title('Fig7: Kernel Config vs Latency (B group)')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig7_kernel_vs_latency.png'))
        plt.close(fig)
        print('  fig7_kernel_vs_latency.png')

    # ---------- 报告 ----------
    print('\n===== 生成报告 =====')
    lines = []
    lines.append('# RK3566 Controlled Benchmark 报告\n')
    lines.append('| 项 | 值 |\n|----|----|')
    lines.append('| 平台 | K11C / RK3566, librknnrt 1.5.2, NPU 900MHz |')
    lines.append('| 输入 | GTSRB 64×64×3, INT8 量化 |')
    lines.append('| Benchmark | warmup=10, iterations=50, 平均值±std |')
    lines.append(f'| 模型数 | {len(df)} |\n')

    lines.append('## 1. 相关性 (vs NPU latency)\n')
    lines.append('| 指标 | Pearson | Spearman |')
    lines.append('|------|--------:|---------:|')
    for name, p, s in corr_rows:
        lines.append(f'| {name} | {p:+.3f} | {s:+.3f} |')

    lines.append('\n## 2. 回归\n')
    lines.append('| 模型 | 公式 | R² | MAE(us) | MAPE(%) |')
    lines.append('|------|------|----:|--------:|--------:|')
    lines.append(f'| M1 | T={coef1[0]:.3f}·MACs{coef1[1]:+.1f} | '
                 f'{r2_1:.3f} | {mae1:.1f} | {mape1:.1f} |')
    lines.append(f'| M2 | T={coef2[0]:.3f}·Conv+{coef2[1]:.3f}·Lin{coef2[2]:+.1f} | '
                 f'{r2_2:.3f} | {mae2:.1f} | {mape2:.1f} |')

    lines.append('\n## 3. Benchmark 数据表\n')
    lines.append(df[['experiment_group', 'model_id', 'independent_var',
                     'params', 'macs', 'conv_macs', 'linear_macs',
                     'flatten_dimension', 'conv_latency_us',
                     'gemm_latency_us', 'npu_latency_us',
                     'mean_latency_us']].to_markdown(index=False))

    lines.append('\n## 4. Findings\n')
    lines.append('- 总 MACs 与 latency 的 Pearson 相关性: '
                 f'{corr_rows[0][1]:+.3f} (当前实验数据)')
    lines.append('- 区分 Conv/GEMM 后回归 R²: '
                 f'{r2_1:.3f} → {r2_2:.3f}')
    lines.append('- Flatten 维度与 GEMM latency 的关系见 Fig5')
    lines.append('- 具体结论以实验组内对比为准 (A/B/C/D 组分别分析)')

    lines.append('\n## 5. Limitations\n')
    lines.append('- 样本量有限, 相关性/回归结果为探索性')
    lines.append('- perf_debug 模式会降低性能, 逐层数据用于相对分析')
    lines.append('- 随机权重模型 (A/B/C/D 组) 的 latency 与权重值无关, '
                 '仅反映结构性能')
    lines.append('- 未覆盖更深/更宽/不同池化策略的完整空间')

    report = os.path.join(BASE_DIR, 'docs',
                          'controlled_benchmark_report.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  {report}')

    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
