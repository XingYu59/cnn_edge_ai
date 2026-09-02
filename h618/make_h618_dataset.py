"""
H618 数据集合并: 静态特征 + 实测延迟 (Phase 6)
==============================================
读 h618_latency.csv (14 模型 CPU/Vulkan), 合并静态特征
(params/macs/conv_macs/linear_macs/flatten) → h618_dataset.csv
静态特征复用 rk3566 的 hpm.architecture (同结构保证双平台可比)。
"""
import os
import sys

import pandas as pd

H618_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H618_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(H618_DIR), 'rk3566'))

from hpm.architecture import analyze_architecture

# 模型配置来源
sys.path.insert(0, os.path.join(os.path.dirname(H618_DIR), 'rk3566'))
from controlled_experiments import ALL_MODELS as EXP
from validation_models import VALIDATION_MODELS

_BASE = {'num_classes': 43, 'input_size': 64}


def get_cfg(model_id):
    if model_id in ('cnn_test', 'd3_k3', 'd5_k3', 'd5_k5'):
        import convert_to_ncnn
        return dict(convert_to_ncnn.MODELS[model_id])
    for m in EXP:
        if m['model_id'] == model_id and m['group'] != 'legacy':
            return dict(_BASE, **m['cfg'])
    for m in VALIDATION_MODELS:
        if m['model_id'] == model_id:
            return dict(_BASE, **m['cfg'])
    return None


def main():
    lat = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_latency.csv'))
    lat['model_id'] = lat['model'].str.replace('models/', '', regex=False)

    cpu = lat[lat['backend'] == 'cpu'][['model_id', 'mean_ms']]
    vk = lat[lat['backend'] == 'vulkan'][['model_id', 'mean_ms']]
    cpu.columns = ['model_id', 'cpu_mean_ms']
    vk.columns = ['model_id', 'vulkan_mean_ms']
    df = cpu.merge(vk, on='model_id')

    rows = []
    for _, r in df.iterrows():
        cfg = get_cfg(r['model_id'])
        p = analyze_architecture(cfg)
        rows.append({
            'model_id': r['model_id'],
            'params': p['params'],
            'total_macs': p['total_macs'],
            'conv_macs': p['conv_macs'],
            'linear_macs': p['linear_macs'],
            'flatten_dim': p['flatten_dim'],
            'depth': p['depth'],
            'channels': str(p['channels']),
            'kernel_size': str(p['kernel_size']),
            'cpu_mean_ms': r['cpu_mean_ms'],
            'vulkan_mean_ms': r['vulkan_mean_ms'],
            'cpu_vulkan_ratio': round(r['cpu_mean_ms'] / r['vulkan_mean_ms'], 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'),
               index=False)
    print(f'已生成 results/h618_dataset.csv ({len(out)} 模型)')
    print(out[['model_id', 'params', 'total_macs', 'flatten_dim',
               'cpu_mean_ms', 'vulkan_mean_ms', 'cpu_vulkan_ratio']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
