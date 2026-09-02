"""
H618 CPU Predictor v2 (25 模型 + 5-fold CV, Phase 10-13)
=========================================================
比较:
  M1: T = a*MACs + b
  M2: T = a*ConvMACs + b*LinearMACs + c
  M3: M2 + flatten 特征
5-fold cross-validation, 输出 R²/MAE/MAPE + cv 方差。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

H618_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(H618_DIR, 'models')


def fit_eval(X, y, coef=None):
    if coef is None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yp = X @ coef
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - yp).mean())
    mape = float((np.abs(y - yp) / np.abs(y)).mean() * 100)
    return coef, r2, mae, mape


def build_X(df, variant):
    if variant == 'M1':
        return np.column_stack([df['total_macs'].values / 1e6,
                                np.ones(len(df))])
    if variant == 'M2':
        return np.column_stack([df['conv_macs'].values / 1e6,
                                df['linear_macs'].values / 1e6,
                                np.ones(len(df))])
    # M3: + flatten/1000
    return np.column_stack([df['conv_macs'].values / 1e6,
                            df['linear_macs'].values / 1e6,
                            df['flatten_dim'].values / 1000.0,
                            np.ones(len(df))])


def main():
    df = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'))
    y = df['cpu_mean_ms'].values

    print(f'===== H618 CPU Predictor v2 ({len(df)} 模型, 5-fold CV) =====')
    rng = np.random.RandomState(42)
    folds = 5
    idx = rng.permutation(len(df))

    results = {}
    for variant in ['M1', 'M2', 'M3']:
        fold_metrics = []
        for k in range(folds):
            va = idx[k::folds]
            tr = np.setdiff1d(np.arange(len(df)), va)
            Xtr = build_X(df.iloc[tr], variant)
            Xva = build_X(df.iloc[va], variant)
            coef, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
            # val metrics
            yp = Xva @ coef
            r2 = 1 - ((y[va] - yp) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum()
            mae = float(np.abs(y[va] - yp).mean())
            mape = float((np.abs(y[va] - yp) / np.abs(y[va])).mean() * 100)
            fold_metrics.append((r2, mae, mape))
        arr = np.array(fold_metrics)
        results[variant] = arr
        print(f'  {variant}: val R²={arr[:,0].mean():.3f}±{arr[:,0].std():.3f} '
              f'MAE={arr[:,1].mean():.2f}±{arr[:,1].std():.2f} '
              f'MAPE={arr[:,2].mean():.1f}±{arr[:,2].std():.1f}%')

    # 选择最佳 (CV MAPE 均值)
    best = min(results, key=lambda v: results[v][:, 2].mean())
    print(f'\n  最佳: {best}')

    # 全量拟合保存
    if best == 'M1':
        coef, r2, mae, mape = fit_eval(build_X(df, best), y)
        model = {'features': ['total_macs'], 'macs_unit': 'M',
                 'coefficients': {'total_macs': round(float(coef[0]), 5)},
                 'intercept': round(float(coef[1]), 4), 'unit': 'ms'}
    elif best == 'M2':
        coef, r2, mae, mape = fit_eval(build_X(df, best), y)
        model = {'features': ['conv_macs', 'linear_macs'], 'macs_unit': 'M',
                 'coefficients': {'conv_macs': round(float(coef[0]), 6),
                                  'linear_macs': round(float(coef[1]), 6)},
                 'intercept': round(float(coef[2]), 4), 'unit': 'ms'}
    else:
        coef, r2, mae, mape = fit_eval(build_X(df, best), y)
        model = {'features': ['conv_macs', 'linear_macs', 'flatten_dim'],
                 'macs_unit': 'M',
                 'coefficients': {'conv_macs': round(float(coef[0]), 6),
                                  'linear_macs': round(float(coef[1]), 6),
                                  'flatten_dim': round(float(coef[2]), 6)},
                 'intercept': round(float(coef[3]), 4), 'unit': 'ms'}

    cv = results[best]
    model.update({
        'hardware': 'H618', 'backend': 'cpu', 'version': 'v2',
        'n_models': len(df),
        'chosen_model': best,
        'all_data': {'r2': round(r2, 4), 'mae_ms': round(mae, 3),
                     'mape': round(mape, 2)},
        'cv5': {'r2_mean': round(float(cv[:, 0].mean()), 4),
                'r2_std': round(float(cv[:, 0].std()), 4),
                'mape_mean': round(float(cv[:, 2].mean()), 2),
                'mape_std': round(float(cv[:, 2].std()), 2)},
        'note': f'25 模型 (14 legacy + 11 扩展), 5-fold CV, '
                f'输入 64x64x3, NCNN CPU',
    })
    out = os.path.join(MODELS_DIR, 'h618_cpu_latency_v2.json')
    with open(out, 'w') as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    print(f'  保存: {out} (选用 {best})')
    print('\n===== 完成 =====')


if __name__ == '__main__':
    main()
