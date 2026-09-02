"""
RK3566 Controlled Benchmark (Task Plan 第二阶段, 第 7/11/12/13 节)
=================================================================
对 controlled_experiments.py 中所有模型执行一致的测量方法:
  warmup=10 + 正式 iterations=50 → mean/min/max/std
  eval_perf → NPU 总时间
  perf_debug → 逐层 latency → conv/gemm/other 分类汇总

输出: results/benchmark_results.csv (统一格式, 含 experiment_group)

用法 (板子已连接):
  python benchmark_rknn3566.py [--group A_depth] [--num 50] [--warmup 10]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from controlled_experiments import ALL_MODELS
from modules.generator import build_model
from modules.analyzer import analyze_model_detail

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
BENCH_CSV = os.path.join(RESULTS_DIR, 'benchmark_results.csv')

# CSV 字段 (Task 第 13 节)
FIELDS = [
    'experiment_group', 'model_id', 'independent_var',
    'depth', 'kernel_config', 'channel_config', 'input_size',
    'params', 'macs', 'conv_macs', 'linear_macs',
    'final_feature_channels', 'final_feature_height', 'final_feature_width',
    'flatten_dimension', 'classifier_params', 'classifier_macs',
    'conv_latency_us', 'gemm_latency_us', 'other_latency_us',
    'total_layer_latency_us', 'npu_latency_us',
    'mean_latency_us', 'std_latency_us', 'min_latency_us', 'max_latency_us',
    'e2e_fps', 'num_runs', 'warmup',
]


def static_analysis(entry: dict) -> dict:
    """统一静态分析 (Task 第 10 节)。"""
    cfg = entry['cfg']
    detail = analyze_model_detail(build_model(cfg))
    s, c = detail['summary'], detail['classifier']
    fm = c['feature_map_shape']
    return {
        'depth': cfg['depth'],
        'kernel_config': str(cfg['kernel_size']),
        'channel_config': str(cfg['channels']),
        'input_size': cfg['input_size'],
        'params': s['parameters'],
        'macs': s['total_macs'],
        'conv_macs': s['conv_macs'],
        'linear_macs': s['linear_macs'],
        'final_feature_channels': fm[0] if fm else None,
        'final_feature_height': fm[1] if fm else None,
        'final_feature_width': fm[2] if fm else None,
        'flatten_dimension': c['flatten_dimension'],
        'classifier_params': c['classifier_parameters'],
        'classifier_macs': c['classifier_macs'],
    }


def measure_latency(rknn_path: str, num: int, warmup: int,
                    target: str = 'rk3566') -> dict:
    """板端测延迟 (Task 第 11 节: 一致流程)。"""
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0, f'load_rknn failed: {rknn_path}'
    assert rknn.init_runtime(target=target) == 0, 'init_runtime failed'

    img = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)

    for _ in range(warmup):
        rknn.inference(inputs=[img], data_format=['nhwc'])

    times = []
    for _ in range(num):
        t0 = time.perf_counter()
        rknn.inference(inputs=[img], data_format=['nhwc'])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    npu_us = None
    try:
        text = str(rknn.eval_perf())
        m = re.search(r'Total Time\(us\):\s*(\d+)', text)
        if m:
            npu_us = int(m.group(1))
    except Exception:
        pass

    rknn.release()
    return {
        'npu_latency_us': npu_us,
        'mean_latency_us': round(float(times.mean() * 1000), 2),
        'std_latency_us': round(float(times.std() * 1000), 2),
        'min_latency_us': round(float(times.min() * 1000), 2),
        'max_latency_us': round(float(times.max() * 1000), 2),
        'e2e_fps': round(float(1000 / times.mean()), 2),
        'num_runs': num,
        'warmup': warmup,
    }


def classify_layer_latency(rknn_path: str) -> dict:
    """perf_debug 逐层解析, 分类汇总 (Task 第 12 节)。"""
    code = (
        'from rknn.api import RKNN\n'
        f'rknn = RKNN(verbose=False)\n'
        f'rknn.load_rknn({rknn_path!r})\n'
        'rknn.init_runtime(target="rk3566", perf_debug=True)\n'
        'rknn.eval_perf()\n'
        'rknn.release()\n'
    )
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True, timeout=120)
    text = proc.stdout + proc.stderr

    conv_us, gemm_us, other_us = 0.0, 0.0, 0.0
    in_table = False
    for line in text.splitlines():
        line = line.replace('\x1b[0m', '').strip()
        if line.startswith('ID') and 'OpType' in line:
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            int(parts[0])
            t_us = float(parts[9])
            dtype = parts[2]
        except ValueError:
            continue
        if 'FLOAT16' in dtype:
            gemm_us += t_us          # FLOAT16 GEMM / Linear 路径
        elif dtype in ('INT8', 'UINT8'):
            conv_us += t_us          # int8 Conv / Pool / 其他 int8 算子
        else:
            other_us += t_us
    return {
        'conv_latency_us': round(conv_us, 1),
        'gemm_latency_us': round(gemm_us, 1),
        'other_latency_us': round(other_us, 1),
        'total_layer_latency_us': round(conv_us + gemm_us + other_us, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', default=None, help='只测某组, 如 A_depth')
    p.add_argument('--num', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()

    models = [m for m in ALL_MODELS
              if args.group is None or m['group'] == args.group]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(BENCH_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for entry in models:
        mid = entry['model_id']
        print(f'\n===== [{entry["group"]}] {mid} ({entry["independent_var"]}) =====')
        stat = static_analysis(entry)
        print(f'  MACs={stat["macs"]/1e6:.1f}M (conv {stat["conv_macs"]/1e6:.1f}M '
              f'+ lin {stat["linear_macs"]/1e6:.1f}M) | '
              f'flatten={stat["flatten_dimension"]}')

        lat = measure_latency(entry['rknn'], args.num, args.warmup)
        layer = classify_layer_latency(entry['rknn'])
        print(f'  mean={lat["mean_latency_us"]/1000:.2f}ms '
              f'±{lat["std_latency_us"]/1000:.2f} | NPU={lat["npu_latency_us"]/1000:.2f}ms')
        print(f'  layer: conv={layer["conv_latency_us"]}us '
              f'gemm={layer["gemm_latency_us"]}us '
              f'other={layer["other_latency_us"]}us')

        row = {
            'experiment_group': entry['group'],
            'model_id': mid,
            'independent_var': entry['independent_var'],
            **stat, **lat, **layer,
        }
        with open(BENCH_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

    print(f'\n===== 完成: {BENCH_CSV} ({len(models)} 个模型) =====')


if __name__ == '__main__':
    main()
