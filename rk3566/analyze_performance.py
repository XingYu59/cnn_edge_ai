"""
性能分析: MACs vs RK3566 latency (Task Plan 第 13/14 节)
=========================================================
读取 results/benchmark_results.csv 与 results/layer_latency.csv,
生成:
  - Benchmark Table (打印)
  - 回归 1: T = a*total_macs + b              (仅总 MACs)
  - 回归 2: T = a*conv_macs + b*linear_macs + c  (结构感知)
  - 4 张散点图 + 1 张逐层耗时分解图 -> docs/figures/

注意 (第 15 节): 数据点仅 4 个, 回归是探索性分析,
不声称已建立准确硬件性能模型。
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

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


def linfit(x, y):
    """一元线性最小二乘: y = a*x + b, 返回 (a, b, R2, MAE, MAPE, y_pred)。"""
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coef
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - y_pred).mean())
    mape = float((np.abs(y - y_pred) / y).mean() * 100)
    return coef[0], coef[1], r2, mae, mape, y_pred


def multilinear_fit(X, y):
    """多元线性最小二乘: y = X @ coef, 返回 (coef, R2, MAE, MAPE)。"""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - y_pred).mean())
    mape = float((np.abs(y - y_pred) / y).mean() * 100)
    return coef, r2, mae, mape, y_pred


def scatter_fit(ax, x, y, labels, xlabel, title, unit='M'):
    """散点 + 线性拟合 + 标注。"""
    xv = x / 1e6 if unit == 'M' else x
    a, b, r2, mae, mape, yp = linfit(xv, y)
    ax.scatter(xv, y, s=60, c='steelblue', zorder=3)
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, a * xs + b, 'r--', lw=1,
            label=f'fit: y={a:.2f}x{b:+.1f} (R²={r2:.3f})')
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (xv[i], y[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('NPU latency (ms)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return r2, mae, mape


def main():
    bench = pd.read_csv(os.path.join(BASE_DIR, 'results', 'benchmark_results.csv'))

    # ---------- 1. Benchmark Table ----------
    print('===== Benchmark Table (RK3566 实测) =====')
    cols = ['model_id', 'parameters', 'total_macs', 'conv_macs',
            'linear_macs', 'feature_map_shape', 'flatten_dimension',
            'classifier_macs', 'npu_latency_ms', 'e2e_latency_ms_mean']
    print(bench[cols].to_string(index=False))
    print()

    y = bench['npu_latency_ms'].values
    labels = bench['model_id'].tolist()

    # ---------- 2. 回归 1: 总 MACs ----------
    print('===== 回归 1: T = a*total_MACs + b =====')
    a1, b1, r2_1, mae1, mape1, _ = linfit(
        bench['total_macs'].values / 1e6, y)
    print(f'  T = {a1:.4f}*MACs(M) {b1:+.3f}')
    print(f'  R²={r2_1:.4f} | MAE={mae1:.4f}ms | MAPE={mape1:.1f}%')
    print()

    # ---------- 3. 回归 2: conv + linear 分开 ----------
    print('===== 回归 2: T = a*ConvMACs + b*LinearMACs + c =====')
    X2 = np.column_stack([bench['conv_macs'].values / 1e6,
                          bench['linear_macs'].values / 1e6,
                          np.ones(len(bench))])
    coef2, r2_2, mae2, mape2, _ = multilinear_fit(X2, y)
    print(f'  T = {coef2[0]:.4f}*ConvM {coef2[1]:.4f}*LinM {coef2[2]:+.3f}')
    print(f'  R²={r2_2:.4f} | MAE={mae2:.4f}ms | MAPE={mape2:.1f}%')
    print()

    # ---------- 4. 回归 3: 总 MACs + flatten 维度 ----------
    print('===== 回归 3: T = a*total_MACs + b*flatten_dim + c =====')
    X3 = np.column_stack([bench['total_macs'].values / 1e6,
                          bench['flatten_dimension'].values / 1000.0,
                          np.ones(len(bench))])
    coef3, r2_3, mae3, mape3, _ = multilinear_fit(X3, y)
    print(f'  T = {coef3[0]:.4f}*MACsM {coef3[1]:.4f}*flattenK {coef3[2]:+.3f}')
    print(f'  R²={r2_3:.4f} | MAE={mae3:.4f}ms | MAPE={mape3:.1f}%')
    print()

    print('===== 模型比较 =====')
    print(f'  {"模型":<12}{"R²":>8}{"MAE(ms)":>10}{"MAPE(%)":>10}')
    print(f'  {"M1 总MACs":<12}{r2_1:>8.3f}{mae1:>10.4f}{mape1:>10.1f}')
    print(f'  {"M2 分算子":<12}{r2_2:>8.3f}{mae2:>10.4f}{mape2:>10.1f}')
    print(f'  {"M3 +flatten":<12}{r2_3:>8.3f}{mae3:>10.4f}{mape3:>10.1f}')
    print()
    print('  [注意] 仅 4 个数据点, 结果为探索性, 不构成最终性能模型 (Task 第15节)')

    # ---------- 5. 图 ----------
    print('\n===== 生成图 -> docs/figures/ =====')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    scatter_fit(axes[0, 0], bench['total_macs'].values, y, labels,
                'Total MACs (M)', 'Fig1: MACs vs Latency')
    scatter_fit(axes[0, 1], bench['parameters'].values, y, labels,
                'Parameters', 'Fig2: Parameters vs Latency', unit='')
    scatter_fit(axes[1, 0], bench['classifier_macs'].values, y, labels,
                'Classifier MACs (M)', 'Fig3: Classifier MACs vs Latency')
    scatter_fit(axes[1, 1], bench['flatten_dimension'].values, y, labels,
                'Flatten Dimension', 'Fig4: Classifier Input Dim vs Latency',
                unit='')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1234_metrics_vs_latency.png'))
    plt.close(fig)
    print('  fig1234_metrics_vs_latency.png')

    # Fig5: 逐层耗时分解 (conv / gemm / other)
    try:
        ll = pd.read_csv(os.path.join(BASE_DIR, 'results', 'layer_latency.csv'))
        models = []
        conv_t, gemm_t, other_t = [], [], []
        for mid in bench['model_id']:
            sub = ll[ll['model_id'] == mid]
            models.append(mid)
            conv_t.append(sub[sub['data_type'] == 'INT8']['time_us'].sum())
            gemm_t.append(sub[sub['data_type'] == 'FLOAT16']['time_us'].sum())
            other_t.append(sub[~sub['data_type'].isin(['INT8', 'FLOAT16'])]['time_us'].sum())
        fig, ax = plt.subplots(figsize=(9, 5))
        xpos = np.arange(len(models))
        ax.bar(xpos, conv_t, label='int8 conv/pool', color='steelblue')
        ax.bar(xpos, gemm_t, bottom=conv_t, label='FLOAT16 gemm/classifier',
               color='coral')
        ax.bar(xpos, other_t, bottom=np.array(conv_t) + np.array(gemm_t),
               label='other', color='lightgray')
        for i, m in enumerate(models):
            ax.text(i, conv_t[i] + gemm_t[i] + other_t[i] + 20, f'{m}',
                    ha='center', fontsize=9)
        ax.set_xticks(xpos)
        ax.set_xticklabels(models)
        ax.set_ylabel('time (us)')
        ax.set_title('Fig5: Layer Latency Breakdown (perf_debug)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig5_layer_latency_breakdown.png'))
        plt.close(fig)
        print('  fig5_layer_latency_breakdown.png')
    except Exception as e:
        print(f'  Fig5 生成失败: {e}')

    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
