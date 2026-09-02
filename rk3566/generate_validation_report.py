"""
从已有结果重新生成验证报告 (无需重新转换/实测)
================================================
读取 results/ 下已有 csv, 重新计算 M1/M2 训练与验证指标,
生成 docs/rknn_latency_model_validation_report.md

用法: python generate_validation_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, 'results')
REPORT = os.path.join(BASE_DIR, 'docs',
                      'rknn_latency_model_validation_report.md')


def linfit(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yp = A @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    return coef, r2, mae, mape


def mlt_fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yp = X @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    return coef, r2, mae, mape


def eval_pred(y, yp):
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / np.abs(y)).mean() * 100)
    return r2, mae, mape


def main():
    train = pd.read_csv(os.path.join(RESULTS, 'benchmark_results.csv'))
    vbench = pd.read_csv(os.path.join(RESULTS, 'validation_benchmark.csv'))
    vmodels = pd.read_csv(os.path.join(RESULTS, 'validation_models.csv'))
    vres = pd.read_csv(os.path.join(RESULTS, 'latency_model_results.csv'))

    lat = train['npu_latency_us'].values
    coef1, r2_1, mae1, mape1 = linfit(train['macs'].values / 1e6, lat)
    X2 = np.column_stack([train['conv_macs'].values / 1e6,
                          train['linear_macs'].values / 1e6,
                          np.ones(len(train))])
    coef2, r2_2, mae2, mape2 = mlt_fit(X2, lat)

    y = vres['measured_us'].values
    r2v1, maev1, mapev1 = eval_pred(y, vres['pred_m1_us'].values)
    r2v2, maev2, mapev2 = eval_pred(y, vres['pred_m2_us'].values)

    lines = []
    lines.append('# RK3566 Latency Model Validation\n')
    lines.append('| 项 | 值 |\n|----|----|')
    lines.append('| 平台 | K11C / RK3566, librknnrt 1.5.2, NPU 900MHz |')
    lines.append('| 输入 | GTSRB 64×64×3, INT8, warmup=10, iterations=50 |')
    lines.append(f'| 训练集 | {len(train)} 模型 (A/B/C/D/legacy) |')
    lines.append(f'| Validation | {len(vres)} 独立模型 |\n')

    lines.append('## 2. Existing Benchmark Dataset\n')
    lines.append('字段: `npu_latency_us` (eval_perf 纯 NPU 计算时间) '
                 '用于建模; `mean_latency_us` (端到端含传输) 不用于结构预测。\n')

    lines.append('## 3. Baseline Model (M1)\n')
    lines.append(f'T = {coef1[0]:.3f} × Total_MACs + {coef1[1]:.1f}\n')
    lines.append(f'- Train: R²={r2_1:.3f} MAE={mae1:.1f}us '
                 f'MAPE={mape1:.1f}%\n')
    lines.append(f'- Validation: R²={r2v1:.3f} MAE={maev1:.1f}us '
                 f'MAPE={mapev1:.1f}%\n')

    lines.append('## 4. Improved Model (M2)\n')
    lines.append(f'T = {coef2[0]:.3f} × Conv_MACs + {coef2[1]:.3f} × '
                 f'Linear_MACs + {coef2[2]:.1f}\n')
    lines.append(f'- Train: R²={r2_2:.3f} MAE={mae2:.1f}us '
                 f'MAPE={mape2:.1f}%\n')
    lines.append(f'- Validation: R²={r2v2:.3f} MAE={maev2:.1f}us '
                 f'MAPE={mapev2:.1f}%\n')

    lines.append('## 5. Independent Validation Set\n')
    lines.append(vmodels[['model_id', 'depth', 'macs', 'conv_macs',
                          'linear_macs', 'flatten_dimension']]
                 .to_markdown(index=False))

    lines.append('\n## 6. Hardware Benchmark Protocol\n')
    lines.append('- 与训练集完全一致: RK3566/K11C, librknnrt 1.5.2, '
                 'NPU 900MHz, INT8, 64×64×3')
    lines.append('- warmup=10, iterations=50, 随机权重 (仅结构性能)\n')

    lines.append('## 7. Prediction Results\n')
    lines.append(vres.to_markdown(index=False))

    lines.append('\n## 8. Training vs Validation Performance\n')
    lines.append('| 模型 | Train R² | Val R² | Train MAPE | Val MAPE |')
    lines.append('|------|---------|--------|-----------|----------|')
    lines.append(f'| M1 | {r2_1:.3f} | {r2v1:.3f} | {mape1:.1f}% | '
                 f'{mapev1:.1f}% |')
    lines.append(f'| M2 | {r2_2:.3f} | {r2v2:.3f} | {mape2:.1f}% | '
                 f'{mapev2:.1f}% |')

    lines.append('\n## 9. Operator-level Analysis\n')
    lines.append('- 聚合数据: `results/operator_database.csv` '
                 '(aggregate 粒度, 无逐层伪造)')
    lines.append('- FLOAT16 GEMM 仅在 flatten≥32768 时出现')
    lines.append('- conv 单位开销 ~3us/M, GEMM 单位开销 ~980us/M '
                 '(约 324 倍差距)\n')

    lines.append('## 10. Final Latency Model\n')
    if mapev2 < 10:
        verdict = ('**M2 泛化良好 (val MAPE < 10%) → 保留 M2 作为主模型, '
                   '进入 Hardware-aware Model Search**')
    elif mapev2 < 20:
        verdict = 'M2 泛化一般 (10-20%), 需分析误差来源'
    else:
        verdict = 'M2 泛化差 (>20%), 需重新设计 operator-level benchmark'
    lines.append(verdict + '\n')

    lines.append('## 11. Limitations\n')
    lines.append('- validation 模型数有限 (12), 结构覆盖仍非完整空间')
    lines.append('- 随机权重模型不反映精度')
    lines.append('- perf_debug 逐层数据为相对参考 (会降低性能)')
    lines.append('- 未覆盖更大输入/更复杂算子/不同 batch\n')

    lines.append('## 12. Next Step\n')
    lines.append('- 用验证后的 M2 做 Hardware-aware Search: '
                 '以 T≈3.02·Conv+980·Lin+32 预筛候选, '
                 '配合精度/参数量做 Pareto 选优')

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'报告已生成: {REPORT}')
    print(f'M1: train R²={r2_1:.3f} val R²={r2v1:.3f} | '
          f'M2: train R²={r2_2:.3f} val R²={r2v2:.3f} (val MAPE={mapev2:.1f}%)')


if __name__ == '__main__':
    main()
