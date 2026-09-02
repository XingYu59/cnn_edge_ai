"""
H618 Latency Predictor (Phase 10)
=================================
基于 h618_dataset.csv 拟合 CPU/Vulkan 延迟模型:
  M1: T = a*MACs + b
  M2: T = a*ConvMACs + b*LinearMACs + c
验证: holdout (10 train / 4 val), 报告 R²/MAE/MAPE
输出: models/h618_cpu_latency_v1.json, h618_vulkan_latency_v1.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

H618_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(H618_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yp = X @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / y).mean() * 100)
    return coef, r2, mae, mape


def eval_pred(y, yp):
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(r2), float(np.abs(y - yp).mean()), \
        float((np.abs(y - yp) / np.abs(y)).mean() * 100)


def build_models(target_col, train, val):
    """拟合 M1/M2, 返回最佳模型 dict + 结果。"""
    ytr = train[target_col].values
    yva = val[target_col].values

    # M1: MACs
    X1tr = np.column_stack([train['total_macs'].values / 1e6,
                            np.ones(len(train))])
    c1, r2_1, mae1, mape1 = fit(X1tr, ytr)
    r2v1, maev1, mapev1 = eval_pred(yva, np.column_stack(
        [val['total_macs'].values / 1e6, np.ones(len(val))]) @ c1)

    # M2: conv + linear
    X2tr = np.column_stack([train['conv_macs'].values / 1e6,
                            train['linear_macs'].values / 1e6,
                            np.ones(len(train))])
    c2, r2_2, mae2, mape2 = fit(X2tr, ytr)
    r2v2, maev2, mapev2 = eval_pred(yva, np.column_stack(
        [val['conv_macs'].values / 1e6,
         val['linear_macs'].values / 1e6,
         np.ones(len(val))]) @ c2)

    print(f'  M1(T=a·MACs+b):  train R²={r2_1:.3f} MAPE={mape1:.1f}% | '
          f'val R²={r2v1:.3f} MAPE={mapev1:.1f}%')
    print(f'  M2(+Linear):      train R²={r2_2:.3f} MAPE={mape2:.1f}% | '
          f'val R²={r2v2:.3f} MAPE={mapev2:.1f}%')
    # 选 val MAPE 更小的
    if mapev2 <= mapev1:
        return {'features': ['conv_macs', 'linear_macs'],
                'macs_unit': 'M',
                'coefficients': {'conv_macs': round(float(c2[0]), 5),
                                 'linear_macs': round(float(c2[1]), 5)},
                'intercept': round(float(c2[2]), 3),
                'unit': 'ms',
                'train': {'r2': round(r2_2, 4), 'mape': round(mape2, 2)},
                'validation': {'r2': round(r2v2, 4),
                               'mae_ms': round(maev2, 3),
                               'mape': round(mapev2, 2)}}, 'M2'
    return {'features': ['total_macs'],
            'macs_unit': 'M',
            'coefficients': {'total_macs': round(float(c1[0]), 5)},
            'intercept': round(float(c1[1]), 3),
            'unit': 'ms',
            'train': {'r2': round(r2_1, 4), 'mape': round(mape1, 2)},
            'validation': {'r2': round(r2v1, 4),
                           'mae_ms': round(maev1, 3),
                           'mape': round(mapev1, 2)}}, 'M1'


def main():
    df = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'))
    # holdout: 按 MACs 排序交错取 4 个做验证 (覆盖范围)
    df = df.sort_values('total_macs').reset_index(drop=True)
    val_idx = [1, 5, 9, 13]   # 分散覆盖
    val = df.iloc[val_idx]
    train = df.drop(val_idx)
    print(f'train={len(train)} val={len(val)}')

    for target, name, out in [
        ('cpu_mean_ms', 'CPU', 'h618_cpu_latency_v1.json'),
        ('vulkan_mean_ms', 'Vulkan', 'h618_vulkan_latency_v1.json')]:
        print(f'\n===== {name} Predictor =====')
        model, chosen = build_models(target, train, val)
        model.update({'hardware': 'H618', 'backend': name.lower(),
                      'chosen_model': chosen,
                      'formula': ('T(ms)=a*ConvMACs(M)+b*LinearMACs(M)+c'
                                  if chosen == 'M2'
                                  else 'T(ms)=a*MACs(M)+b'),
                      'note': f'NCNN {name}, {len(df)} 模型, '
                              f'warmup=20 iters=200, 输入 64x64x3'})
        with open(os.path.join(MODELS_DIR, out), 'w') as f:
            json.dump(model, f, indent=2, ensure_ascii=False)
        print(f'  保存: {out} (选用 {chosen})')

    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
