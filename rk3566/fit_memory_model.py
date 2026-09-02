"""
Memory 校准模型 (Phase 9/10)
============================
基于实测 (results/memory_benchmark.csv), 校准估算内存:

  发现: 估算系统性低估, 且偏差与 flatten 阈值分组:
    flatten ≤ 16384 (int8 路径)      : weight 误差 <1%
    flatten ≥ 32768 (FLOAT16 GEMM)   : weight 低估 ~1.85x (fp16 classifier 权重)

校准策略:
  1. 分段: 按 flatten 阈值分组拟合 meas = a*est + b
  2. 输出 models/rk3566_memory_v1.json
  3. hpm/memory.py 加载校准后输出 predicted_runtime_memory

注意: 10 个样本的初步校准, 不声称最终硬件模型 (Phase 9 谨慎原则)。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

JSON_PATH = os.path.join(BASE_DIR, 'models', 'rknn_memory_v1.json')
FP16_FLATTEN_THRESHOLD = 16384   # FINDING-001: 超过则 FLOAT16 GEMM


def fit_group(df, name):
    """对一组 (est, meas) 拟合 meas = a*est + b。"""
    x = df['est_total_bytes'].values
    y = df['meas_total_bytes'].values
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yp = A @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    print(f'  [{name}] meas = {coef[0]:.4f}*est {coef[1]:+.0f} | '
          f'R²={r2:.3f} MAE={mae:,.0f}B MAPE={mape:.1f}%')
    return {'a': float(coef[0]), 'b': float(coef[1]), 'n': len(df),
            'r2': r2, 'mae': mae, 'mape': mape}


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'results', 'memory_benchmark.csv'))

    # 需要 flatten 信息 (从 memory_profiles.csv 或重新算)
    prof = pd.read_csv(os.path.join(BASE_DIR, 'results',
                                    'memory_profiles.csv'))
    prof = prof[prof['precision'] == 'int8'][['model_id', 'flatten_dim']]
    df = df.merge(prof, on='model_id')

    int8_grp = df[df['flatten_dim'] <= FP16_FLATTEN_THRESHOLD]
    fp16_grp = df[df['flatten_dim'] > FP16_FLATTEN_THRESHOLD]
    print(f'int8 组: {len(int8_grp)} 模型 | fp16 组: {len(fp16_grp)} 模型')

    cal = {}
    cal['int8_path'] = fit_group(int8_grp, 'int8')
    cal['fp16gemm_path'] = fit_group(fp16_grp, 'fp16gemm')

    data = {
        'hardware': 'RK3566',
        'model_type': 'piecewise_linear_calibration',
        'unit': 'bytes',
        'flatten_threshold': FP16_FLATTEN_THRESHOLD,
        'formula': ('M_runtime ≈ a*M_estimated_peak + b, '
                    '按 flatten 阈值分组 (int8/fp16-gemm 路径)'),
        'calibration': cal,
        'note': ('基于 10 个代表模型的初步校准 (Phase 9 情况 B); '
                 'weight 在 int8 路径误差 <1%, fp16-gemm 路径低估 '
                 '由 FLOAT16 classifier 权重导致'),
        'training': {'n_models': len(df)},
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f'\n已导出: {JSON_PATH}')


if __name__ == '__main__':
    main()
