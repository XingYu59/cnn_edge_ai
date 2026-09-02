"""
生成 models/README.md: 所有已测试模型说明文档
=============================================
从实验配置 (controlled_experiments / validation_models) 与实测数据
(benchmark_results / validation_benchmark / memory_benchmark / results)
自动生成每个模型的说明: 来源、配置、用途、实测 latency / memory / 精度。

用法: python gen_models_readme.py
"""
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from controlled_experiments import ALL_MODELS as EXP_MODELS
from validation_models import VALIDATION_MODELS

RESULTS = os.path.join(BASE_DIR, 'results')


def load_latency():
    d = {}
    b = os.path.join(RESULTS, 'benchmark_results.csv')
    if os.path.isfile(b):
        df = pd.read_csv(b)
        d.update(dict(zip(df['model_id'], df['npu_latency_us'])))
    v = os.path.join(RESULTS, 'validation_benchmark.csv')
    if os.path.isfile(v):
        df = pd.read_csv(v)
        d.update(dict(zip(df['model_id'], df['npu_latency_us'])))
    return d


def load_memory():
    d = {}
    p = os.path.join(RESULTS, 'memory_benchmark.csv')
    if os.path.isfile(p):
        df = pd.read_csv(p)
        d.update(dict(zip(df['model_id'],
                          df['meas_total_bytes'] / 1024 / 1024)))
    return d


def load_accuracy():
    d = {}
    p = os.path.join(RESULTS, 'results.csv')
    if os.path.isfile(p):
        df = pd.read_csv(p)
        d.update(dict(zip(df['model_id'], df['test_accuracy'])))
    return d


def cfg_str(cfg):
    ks = cfg['kernel_size']
    k = ks if isinstance(ks, int) else '-'.join(str(x) for x in ks)
    pool = cfg.get('pool_positions', 'auto(pool_every=2)')
    return (f'depth={cfg["depth"]}, ch={cfg["channels"]}, '
            f'k={k}, pool={pool}')


def main():
    lat = load_latency()
    mem = load_memory()
    acc = load_accuracy()

    lines = []
    lines.append('# Models 说明文档\n')
    lines.append('本目录所有 `.rknn` 模型的配置、来源与实测数据。'
                 '实测平台: K11C / RK3566, NPU 900MHz, INT8, 输入 64×64×3。\n')
    lines.append('| 文件 | 来源 | 配置 | 实测延迟 | 实测内存 | 精度 |')
    lines.append('|------|------|------|---------|---------|------|')

    rows = []

    # ---- 1. 训练模型 (trained) ----
    # 文件名与精度名映射 (results.csv 用完整搜索名)
    ACC_NAME = {
        'd3_k3': 'cnn_d3_c16-32-32_k3',
        'cnn_test': 'cnn_test',
        'd5_k3': 'cnn_d5_c32-32-64-64-128_k3',
        'd5_k5': 'cnn_d5_c32-32-64-64-128_k5',
    }
    trained = [m for m in EXP_MODELS if m['group'] == 'legacy']
    for m in sorted(trained, key=lambda x: x['model_id']):
        mid = m['model_id']
        fname = 'cnn_test.rknn' if mid == 'cnn_test' else f'cnn_{mid}.rknn'
        ac = acc.get(ACC_NAME.get(mid, mid))
        rows.append((fname, '**训练模型**', cfg_str(m['cfg']),
                     lat.get(mid), mem.get(mid), ac))

    # ---- 2. 控制变量实验 (exp_*) ----
    for m in sorted(EXP_MODELS, key=lambda x: x['model_id']):
        if m['group'] == 'legacy':
            continue
        mid = m['model_id']
        fname = f'exp_{mid}.rknn'
        group_desc = {
            'A_depth': 'A: depth 实验',
            'B_kernel': 'B: kernel 实验',
            'C_featuremap': 'C: feature map 实验',
            'D_classifier': 'D: classifier 实验',
        }.get(m['group'], m['group'])
        rows.append((fname, group_desc, cfg_str(m['cfg']),
                     lat.get(mid), mem.get(mid), acc.get(mid)))

    # ---- 3. 验证模型 (val_*) ----
    for m in VALIDATION_MODELS:
        mid = m['model_id']
        fname = f'val_{mid}.rknn'
        rows.append((fname, '验证集(独立)', cfg_str(m['cfg']),
                     lat.get(mid), mem.get(mid), acc.get(mid)))

    for fname, src, cfg, lt, mm, ac in rows:
        lt_s = f'{lt/1000:.2f} ms' if lt is not None else '-'
        mm_s = f'{mm:.2f} MB' if mm is not None else '-'
        ac_s = f'{ac*100:.1f}%' if ac is not None else '-'
        lines.append(f'| `{fname}` | {src} | {cfg} | {lt_s} | {mm_s} | {ac_s} |')

    lines.append('\n---\n')
    lines.append('## 分类说明\n')
    lines.append('- **训练模型** (4 个): GTSRB 完整训练, 有真实精度, '
                 '用于精度/效率分析 (benchmark_results.csv 的 legacy 组)')
    lines.append('- **控制变量实验** (exp_*, 随机权重): 第二阶段控制变量实验, '
                 '仅测 latency (与权重值无关), 分 A depth / B kernel / '
                 'C feature map / D classifier 四组')
    lines.append('- **验证集** (val_*, 随机权重): 第三阶段独立验证集, '
                 '用于验证 latency 模型泛化能力 (未参与拟合)')
    lines.append('- **JSON 模型**: `rknn_latency_v1.json` (延迟回归参数) / '
                 '`rknn_memory_v1.json` (内存校准参数)')
    lines.append('\n> 精度 = 训练时 test accuracy (仅训练模型有); '
                 '随机权重模型无精度 (只测结构性能)。\n')

    out = os.path.join(BASE_DIR, 'models', 'README.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'已生成: {out} ({len(rows)} 个模型)')


if __name__ == '__main__':
    main()
