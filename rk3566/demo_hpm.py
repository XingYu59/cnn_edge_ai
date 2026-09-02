"""
HPM 验证与演示 (Task 5/8/10)
============================
1. 验证: 对 12 个 validation 模型, 用 hpm 预测 latency, 与实测对比
   (确认 R²≈0.972, MAE≈112.8us, MAPE≈9.2%, 与已有验证结果一致)
2. 演示: evaluate_candidate + 硬件约束过滤

用法: python demo_hpm.py
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from hpm.architecture import analyze_architecture
from hpm.latency import RKNNLatencyModel
from hpm.pipeline import evaluate_candidate
from hpm.filter import check_hardware_constraints
from validation_models import VALIDATION_MODELS


def main():
    print('===== 1. Predictor 验证 (12 validation 模型) =====')
    model = RKNNLatencyModel.load()
    vbench = pd.read_csv(os.path.join(BASE_DIR, 'results',
                                      'validation_benchmark.csv'))
    vbench = vbench.set_index('model_id')

    rows = []
    for vm in VALIDATION_MODELS:
        mid = vm['model_id']
        profile = analyze_architecture(vm['cfg'])
        pred = model.predict(profile)
        meas = vbench.loc[mid, 'npu_latency_us']
        rows.append({
            'model_id': mid,
            'measured_us': meas,
            'predicted_us': round(pred, 1),
            'abs_err_us': round(abs(pred - meas), 1),
            'rel_err_pct': round((pred - meas) / meas * 100, 1),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    y = res['measured_us'].values
    yp = res['predicted_us'].values
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / np.abs(y)).mean() * 100)
    print(f'\n  Validation: R²={r2:.4f} MAE={mae:.1f}us MAPE={mape:.1f}%')
    print('  (与实验报告一致: R²=0.972, MAE=112.8, MAPE=9.2%)')

    print('\n===== 2. evaluate_candidate 演示 =====')
    demo_configs = [
        {'name': 'light', 'depth': 3, 'channels': [16, 32, 32],
         'kernel_size': 3, 'num_classes': 43, 'input_size': 64},
        {'name': 'medium', 'depth': 4, 'channels': [32, 32, 64, 64],
         'kernel_size': 3, 'num_classes': 43, 'input_size': 64},
        {'name': 'heavy', 'depth': 4, 'channels': [32, 64, 64, 128],
         'kernel_size': 5, 'pool_positions': [2, 4],
         'num_classes': 43, 'input_size': 64},
    ]
    constraints = {
        'max_latency_us': 1500,
        'max_memory_bytes': 8 * 1024 * 1024,   # 8 MB
        'max_params': 2_000_000,
    }
    for cfg in demo_configs:
        r = evaluate_candidate(cfg, constraints=constraints,
                               latency_model=model)
        print(f'  {cfg["name"]:<8} latency={r["predicted_latency_us"]:>6.0f}us '
              f'mem={r["estimated_memory_bytes"]/1024/1024:.2f}MB '
              f'params={r["params"]:,} feasible={r["feasible"]} '
              f'{r["violations"] if not r["feasible"] else ""}')

    print('\n===== 3. 直接 filter 演示 =====')
    p = analyze_architecture(demo_configs[2])
    chk = check_hardware_constraints(p, constraints, model)
    print(f'  heavy 模型: feasible={chk["feasible"]}, '
          f'violations={chk["violations"]}')


if __name__ == '__main__':
    main()
