"""
RK3566 Latency Model (Task 3/4)
===============================
封装实验得到的线性回归模型:
    T(us) = a * Conv_MACs(M) + b * Linear_MACs(M) + c

模型参数持久化在 models/rknn_latency_v1.json, 由真实实验数据拟合 (禁止手猜)。
"""
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rknn_latency_v1.json')


class RKNNLatencyModel:
    """RK3566 latency 线性回归模型。"""

    def __init__(self, data: Dict):
        self.data = data
        self.hardware = data['hardware']
        self.coefficients = data['coefficients']
        self.intercept = data['intercept']
        self.macs_unit = data.get('macs_unit', 'M')

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> 'RKNNLatencyModel':
        if not os.path.isfile(path):
            raise FileNotFoundError(f'latency 模型不存在: {path}')
        with open(path) as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    def predict(self, profile: Dict) -> float:
        """输入 architecture profile, 输出预测 latency (us)。"""
        conv_m = profile['conv_macs'] / 1e6      # M 单位
        lin_m = profile['linear_macs'] / 1e6
        return (self.coefficients['conv_macs'] * conv_m
                + self.coefficients['linear_macs'] * lin_m
                + self.intercept)

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_m2(df: pd.DataFrame):
        """用 CSV 数据拟合 T = a*Conv + b*Linear + c, 返回 (coef, r2, mae, mape)。"""
        y = df['npu_latency_us'].values
        X = np.column_stack([df['conv_macs'].values / 1e6,
                             df['linear_macs'].values / 1e6,
                             np.ones(len(df))])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yp = X @ coef
        r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mae = float(np.abs(y - yp).mean())
        mape = float((np.abs(y - yp) / y).mean() * 100)
        return coef, r2, mae, mape

    # ------------------------------------------------------------------
    @classmethod
    def fit_and_export(cls, train_csv: str, val_results_csv: str,
                       out_path: str = DEFAULT_MODEL_PATH) -> 'RKNNLatencyModel':
        """
        从真实实验数据拟合并导出模型 json (Task 3)。
          train_csv       : results/benchmark_results.csv (22 模型)
          val_results_csv : results/latency_model_results.csv (12 验证模型)
        """
        train = pd.read_csv(train_csv)
        coef, r2_tr, mae_tr, mape_tr = cls._fit_m2(train)

        # validation 指标 (M2 预测 vs 实测)
        vres = pd.read_csv(val_results_csv)
        yv = vres['measured_us'].values
        ypv = vres['pred_m2_us'].values
        r2_val = 1 - ((yv - ypv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        mae_val = float(np.abs(yv - ypv).mean())
        mape_val = float((np.abs(yv - ypv) / np.abs(yv)).mean() * 100)

        data = {
            'hardware': 'RK3566',
            'model_type': 'linear_regression',
            'features': ['conv_macs', 'linear_macs'],
            'macs_unit': 'M',
            'coefficients': {
                'conv_macs': round(float(coef[0]), 4),
                'linear_macs': round(float(coef[1]), 4),
            },
            'intercept': round(float(coef[2]), 2),
            'unit': 'us',
            'formula': 'T(us) = a*ConvMACs(M) + b*LinearMACs(M) + c',
            'training': {
                'n_models': len(train),
                'r2': round(r2_tr, 4),
                'mae_us': round(mae_tr, 2),
                'mape': round(mape_tr, 2),
            },
            'validation': {
                'n_models': len(vres),
                'r2': round(r2_val, 4),
                'mae_us': round(mae_val, 2),
                'mape': round(mape_val, 2),
            },
            'note': ('线性回归性能模型 (非神经网络); '
                     'latency 为 eval_perf 纯 NPU 计算时间; '
                     'accuracy 未纳入, 由独立训练流程提供'),
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'[latency] 模型已导出: {out_path}')
        print(f'  T = {coef[0]:.3f}*Conv + {coef[1]:.3f}*Lin + {coef[2]:.1f}')
        print(f'  train: R²={r2_tr:.3f} MAPE={mape_tr:.1f}% | '
              f'val: R²={r2_val:.3f} MAPE={mape_val:.1f}%')
        return cls(data)


def predict_latency(profile: Dict,
                    model: Optional[RKNNLatencyModel] = None) -> float:
    """便捷函数: 给定 architecture profile 返回预测 latency (us)。"""
    if model is None:
        model = RKNNLatencyModel.load()
    return model.predict(profile)


if __name__ == '__main__':
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == 'fit':
        RKNNLatencyModel.fit_and_export(
            os.path.join(BASE_DIR, 'results', 'benchmark_results.csv'),
            os.path.join(BASE_DIR, 'results', 'latency_model_results.csv'))
    else:
        m = RKNNLatencyModel.load()
        print(json.dumps(m.data, indent=2, ensure_ascii=False))
