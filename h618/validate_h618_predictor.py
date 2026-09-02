"""
Predictor 验证图 (predicted vs measured)
=======================================
加载 h618 模型 json, 对全部 14 模型计算预测, 画散点图。
"""
import json
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


def predict(model, row):
    c = model['coefficients']
    if 'total_macs' in c:
        return c['total_macs'] * row['total_macs'] / 1e6 + model['intercept']
    return (c['conv_macs'] * row['conv_macs'] / 1e6
            + c['linear_macs'] * row['linear_macs'] / 1e6
            + model['intercept'])


def main():
    df = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'))
    for name, target, jf in [
        ('CPU', 'cpu_mean_ms', 'h618_cpu_latency_v1.json'),
        ('Vulkan', 'vulkan_mean_ms', 'h618_vulkan_latency_v1.json')]:
        with open(os.path.join(H618_DIR, 'models', jf)) as f:
            model = json.load(f)
        df[f'pred_{name.lower()}'] = df.apply(
            lambda r: predict(model, r), axis=1)
        y = df[target].values
        yp = df[f'pred_{name.lower()}'].values
        r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mape = float((np.abs(y - yp) / np.abs(y)).mean() * 100)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(y, yp, s=55, c='steelblue', zorder=3)
        lim = [0, max(y.max(), yp.max()) * 1.05]
        ax.plot(lim, lim, 'r--', lw=1, label='y=x')
        for _, r in df.iterrows():
            ax.annotate(r['model_id'], (r[target], r[f'pred_{name.lower()}']),
                        fontsize=7, textcoords='offset points', xytext=(4, 4))
        ax.set_xlabel(f'Measured {name} latency (ms)')
        ax.set_ylabel(f'Predicted {name} latency (ms)')
        ax.set_title(f'H618 {name}: Predicted vs Measured\n'
                     f'(all-data R²={r2:.3f}, MAPE={mape:.1f}%)')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(
            FIG_DIR, f'h618_{name.lower()}_predicted_vs_measured.png'))
        plt.close(fig)
        print(f'  {name}: all-data R²={r2:.3f} MAPE={mape:.1f}%')
    print('===== 完成 =====')


if __name__ == '__main__':
    main()
