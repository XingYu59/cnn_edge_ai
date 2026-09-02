"""
RK3566 Latency Model Validation (Task Plan 第三阶段)
====================================================
流程:
  1. 用训练集 (22 模型, benchmark_results.csv) 拟合 M1/M2
  2. 12 个独立 validation 模型静态分析 -> validation_models.csv
  3. 转换 (随机权重) + RK3566 实测 (warmup10+50) -> validation_benchmark.csv
  4. M1/M2 预测 validation latency, 对比实测 -> latency_model_results.csv
  5. 图: Fig V1/V2/V3
  6. operator_database.csv (conv/gemm 聚合数据)
  7. 报告 rknn_latency_model_validation_report.md + Q1-Q6

用法 (板子已连接, 后台运行约 30-40 分钟):
  python validate_latency_model.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from validation_models import VALIDATION_MODELS
from benchmark_rknn3566 import (static_analysis as bench_static,
                                measure_latency, classify_layer_latency)
from convert_to_rknn import convert_to_rknn
from modules.generator import build_model
from modules.analyzer import analyze_model_detail

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120


# ---------------- 回归工具 ----------------
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


def eval_pred(y, yp):
    """对预测结果计算 R²/MAE/MAPE/Pearson。"""
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / np.abs(y)).mean() * 100)
    pear = float(np.corrcoef(y, yp)[0, 1]) if y.std() and yp.std() else np.nan
    return r2, mae, mape, pear


# ---------------- 主流程 ----------------
def main():
    # ========== 1. 拟合 M1/M2 (只用训练集) ==========
    train = pd.read_csv(os.path.join(RESULTS_DIR, 'benchmark_results.csv'))
    lat = train['npu_latency_us'].values

    coef1, r2_1, mae1, mape1 = linfit(train['macs'].values / 1e6, lat)
    print(f'[M1] T = {coef1[0]:.3f}*MACs + {coef1[1]:.1f} | '
          f'R²={r2_1:.3f} MAE={mae1:.1f}us MAPE={mape1:.1f}%')

    X2 = np.column_stack([train['conv_macs'].values / 1e6,
                          train['linear_macs'].values / 1e6,
                          np.ones(len(train))])
    coef2, r2_2, mae2, mape2 = mlt_fit(X2, lat)
    print(f'[M2] T = {coef2[0]:.3f}*Conv + {coef2[1]:.3f}*Lin + {coef2[2]:.1f} | '
          f'R²={r2_2:.3f} MAE={mae2:.1f}us MAPE={mape2:.1f}%')

    # ========== 2. validation 静态分析 + 转换 + 实测 + 预测 ==========
    vstat_rows, vbench_rows, vres_rows = [], [], []
    for i, vm in enumerate(VALIDATION_MODELS):
        mid = vm['model_id']
        print(f'\n===== validation {mid} ({i+1}/{len(VALIDATION_MODELS)}) =====')

        # 静态分析
        stat = bench_static(vm)
        vstat_rows.append({'model_id': mid, **stat})
        print(f'  MACs={stat["macs"]/1e6:.1f}M (conv {stat["conv_macs"]/1e6:.1f}M '
              f'+ lin {stat["linear_macs"]/1e6:.1f}M) | '
              f'flatten={stat["flatten_dimension"]}')

        # 转换 (随机权重)
        out = os.path.join(BASE_DIR, 'models', f'val_{mid}.rknn')
        if not os.path.isfile(out):
            print(f'  [convert] {mid}...')
            convert_to_rknn(None, vm['cfg'], out, 'rk3566', calib_num=100)

        # 实测
        lat_m = measure_latency(out, num=50, warmup=10)
        layer = classify_layer_latency(out)
        vbench_rows.append({'model_id': mid, **stat, **lat_m, **layer})
        print(f'  mean={lat_m["mean_latency_us"]/1000:.2f}ms '
              f'NPU={lat_m["npu_latency_us"]/1000:.2f}ms | '
              f'conv={layer["conv_latency_us"]}us gemm={layer["gemm_latency_us"]}us')

        # 预测
        y = lat_m['npu_latency_us']
        p1 = coef1[0] * stat['macs'] / 1e6 + coef1[1]
        p2 = (coef2[0] * stat['conv_macs'] / 1e6
              + coef2[1] * stat['linear_macs'] / 1e6 + coef2[2])
        vres_rows.append({
            'model_id': mid,
            'measured_us': y,
            'pred_m1_us': round(float(p1), 1),
            'err_m1_us': round(float(p1 - y), 1),
            'err_m1_pct': round(float((p1 - y) / y * 100), 1),
            'pred_m2_us': round(float(p2), 1),
            'err_m2_us': round(float(p2 - y), 1),
            'err_m2_pct': round(float((p2 - y) / y * 100), 1),
        })
        print(f'  M1 pred={p1:.0f}us (err {(p1-y)/y*100:+.1f}%) | '
              f'M2 pred={p2:.0f}us (err {(p2-y)/y*100:+.1f}%)')

    # 保存 CSV
    pd.DataFrame(vstat_rows).to_csv(
        os.path.join(RESULTS_DIR, 'validation_models.csv'), index=False)
    pd.DataFrame(vbench_rows).to_csv(
        os.path.join(RESULTS_DIR, 'validation_benchmark.csv'), index=False)
    res = pd.DataFrame(vres_rows)
    res.to_csv(os.path.join(RESULTS_DIR, 'latency_model_results.csv'),
               index=False)

    # ========== 3. validation 评估 ==========
    y = res['measured_us'].values
    print('\n===== Validation Performance =====')
    r2v1, maev1, mapev1, pv1 = eval_pred(y, res['pred_m1_us'].values)
    r2v2, maev2, mapev2, pv2 = eval_pred(y, res['pred_m2_us'].values)
    print(f'  M1 validation: R²={r2v1:.3f} MAE={maev1:.1f}us MAPE={mapev1:.1f}%')
    print(f'  M2 validation: R²={r2v2:.3f} MAE={maev2:.1f}us MAPE={mapev2:.1f}%')

    # ========== 4. 图 ==========
    print('\n===== 图 =====')
    # Fig V1: measured vs predicted (M1 + M2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (pcol, name) in zip(axes, [('pred_m1_us', 'M1 (Total MACs)'),
                                       ('pred_m2_us', 'M2 (Conv+GEMM)')]):
        pp = res[pcol].values
        ax.scatter(y, pp, s=45, c='steelblue', zorder=3)
        lim = [min(y.min(), pp.min()) * 0.9, max(y.max(), pp.max()) * 1.1]
        ax.plot(lim, lim, 'r--', lw=1, label='y = x')
        for _, r in res.iterrows():
            ax.annotate(r['model_id'], (r['measured_us'], r[pcol]),
                        textcoords='offset points', xytext=(3, 3), fontsize=6)
        r2v, _, mapev, _ = eval_pred(y, pp)
        ax.set_xlabel('Measured latency (us)')
        ax.set_ylabel('Predicted latency (us)')
        ax.set_title(f'Fig V1: {name} (val R²={r2v:.3f}, MAPE={mapev:.1f}%)')
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_validation_prediction.png'))
    plt.close(fig)
    print('  fig_validation_prediction.png')

    # Fig V2: error vs model id
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xpos = np.arange(len(res))
    ax.bar(xpos - 0.2, res['err_m1_pct'], width=0.4, label='M1 error %',
           color='#d62728', alpha=0.8)
    ax.bar(xpos + 0.2, res['err_m2_pct'], width=0.4, label='M2 error %',
           color='#1f77b4', alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(res['model_id'], fontsize=8)
    ax.set_ylabel('Prediction error (%)')
    ax.set_title('Fig V2: Prediction Error by Model')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_prediction_error.png'))
    plt.close(fig)
    print('  fig_prediction_error.png')

    # Fig V3: M2 单独
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y, res['pred_m2_us'].values, s=55, c='#2ca02c', zorder=3)
    lim = [min(y.min(), res['pred_m2_us'].min()) * 0.9,
           max(y.max(), res['pred_m2_us'].max()) * 1.1]
    ax.plot(lim, lim, 'r--', lw=1, label='y = x')
    for _, r in res.iterrows():
        ax.annotate(r['model_id'], (r['measured_us'], r['pred_m2_us']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xlabel('Measured latency (us)')
    ax.set_ylabel('Predicted latency (us)')
    ax.set_title(f'Fig V3: M2 Validation\nR²={r2v2:.3f} MAE={maev2:.1f}us '
                 f'MAPE={mapev2:.1f}%')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_m2_validation.png'))
    plt.close(fig)
    print('  fig_m2_validation.png')

    # ========== 5. operator_database.csv ==========
    print('\n===== Operator Database =====')
    all_df = pd.concat([train, pd.DataFrame(vbench_rows)], ignore_index=True)
    ops = []
    for _, r in all_df.iterrows():
        if r['conv_latency_us'] > 0:
            ops.append({
                'operator_type': 'conv_aggregate',
                'model_id': r['model_id'],
                'macs': r['conv_macs'],
                'flatten_dimension': r['flatten_dimension'],
                'measured_latency_us': r['conv_latency_us'],
                'granularity': 'aggregate_per_model',
            })
        if r['gemm_latency_us'] > 0:
            ops.append({
                'operator_type': 'gemm_aggregate',
                'model_id': r['model_id'],
                'macs': r['linear_macs'],
                'flatten_dimension': r['flatten_dimension'],
                'measured_latency_us': r['gemm_latency_us'],
                'granularity': 'aggregate_per_model',
            })
    pd.DataFrame(ops).to_csv(
        os.path.join(RESULTS_DIR, 'operator_database.csv'), index=False)
    print(f'  operator_database.csv: {len(ops)} 行 (aggregate 粒度)')

    # ========== 6. 报告 ==========
    print('\n===== 报告 =====')
    lines = []
    lines.append('# RK3566 Latency Model Validation\n')
    lines.append('| 项 | 值 |\n|----|----|')
    lines.append('| 平台 | K11C / RK3566, librknnrt 1.5.2, NPU 900MHz |')
    lines.append('| 输入 | GTSRB 64×64×3, INT8, warmup=10, iterations=50 |')
    lines.append(f'| 训练集 | {len(train)} 模型 |')
    lines.append(f'| Validation | {len(res)} 模型 |\n')

    lines.append('## 3. Baseline Model (M1)\n')
    lines.append(f'T = {coef1[0]:.3f} × MACs + {coef1[1]:.1f}\n')
    lines.append(f'- Train: R²={r2_1:.3f} MAE={mae1:.1f}us MAPE={mape1:.1f}%\n')
    lines.append(f'- Validation: R²={r2v1:.3f} MAE={maev1:.1f}us '
                 f'MAPE={mapev1:.1f}%\n')

    lines.append('## 4. Improved Model (M2)\n')
    lines.append(f'T = {coef2[0]:.3f} × ConvMACs + {coef2[1]:.3f} × '
                 f'LinearMACs + {coef2[2]:.1f}\n')
    lines.append(f'- Train: R²={r2_2:.3f} MAE={mae2:.1f}us MAPE={mape2:.1f}%\n')
    lines.append(f'- Validation: R²={r2v2:.3f} MAE={maev2:.1f}us '
                 f'MAPE={mapev2:.1f}%\n')

    lines.append('## 5. Independent Validation Set\n')
    lines.append(pd.DataFrame(vstat_rows)[['model_id', 'depth', 'macs',
                                           'conv_macs', 'linear_macs',
                                           'flatten_dimension']]
                 .to_markdown(index=False))

    lines.append('\n## 7. Prediction Results\n')
    lines.append(res.to_markdown(index=False))

    lines.append('\n## 8. Training vs Validation\n')
    lines.append('| 模型 | Train R² | Val R² | Train MAPE | Val MAPE |')
    lines.append('|------|---------|--------|-----------|----------|')
    lines.append(f'| M1 | {r2_1:.3f} | {r2v1:.3f} | {mape1:.1f}% | '
                 f'{mapev1:.1f}% |')
    lines.append(f'| M2 | {r2_2:.3f} | {r2v2:.3f} | {mape2:.1f}% | '
                 f'{mapev2:.1f}% |')

    lines.append('\n## 9. Operator-level Analysis\n')
    lines.append('- conv/gemm 聚合数据见 `operator_database.csv` '
                 '(aggregate 粒度, 无逐层伪造)')
    lines.append('- 仅 flatten≥32768 的模型出现 FLOAT16 GEMM 耗时')

    lines.append('\n## 10. Final Latency Model\n')
    if mapev2 < 10:
        verdict = 'M2 泛化良好 (MAPE<10%), 保留为主模型'
    elif mapev2 < 20:
        verdict = 'M2 泛化一般 (MAPE 10-20%), 需分析误差来源'
    else:
        verdict = 'M2 泛化差 (MAPE>20%), 需重新设计 operator-level benchmark'
    lines.append(f'**Verdict: {verdict}**\n')

    lines.append('## 11. Limitations\n')
    lines.append('- validation 模型数与结构覆盖有限')
    lines.append('- 随机权重模型不反映精度, 仅结构性能')
    lines.append('- perf_debug 逐层数据为相对参考')
    lines.append('- 未覆盖更深/更大输入/不同算子')

    lines.append('\n## 12. Next Step\n')
    lines.append('- 根据 verdict: 保留 M2 进入 Hardware-aware Search / '
                 '补充 operator 数据')

    report = os.path.join(BASE_DIR, 'docs',
                          'rknn_latency_model_validation_report.md')
    with open(report, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  {report}')
    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
