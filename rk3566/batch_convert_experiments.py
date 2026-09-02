"""
批量转换实验模型为 rknn (Task Plan 第二阶段)
============================================
对 controlled_experiments.py 中所有未训练的新模型, 用随机权重转换 (仅测 latency)。
相同配置自动去重 (B1=C2=D2, C1=D4 等), 并行转换加速。

用法:
  python batch_convert_experiments.py [--workers 4] [--only A_depth]
"""
import argparse
import multiprocessing as mp
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from controlled_experiments import ALL_MODELS
from convert_to_rknn import convert_to_rknn

MODELS_DIR = os.path.join(BASE_DIR, 'models')


def convert_one(job):
    model_id, cfg, out_path = job
    log_path = os.path.join(BASE_DIR, 'logs', f'convert_{model_id}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # 转换日志重定向到文件
    import contextlib
    with open(log_path, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                convert_to_rknn(None, cfg, out_path, 'rk3566', calib_num=100)
                return (model_id, True, '')
            except Exception as e:
                return (model_id, False, str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--only', default=None, help='只转换某组, 如 A_depth')
    args = p.parse_args()

    # 收集需要转换的模型 (未训练), 按 cfg 去重
    jobs, seen = [], set()
    for m in ALL_MODELS:
        if m['trained']:
            continue
        if args.only and m['group'] != args.only:
            continue
        key = repr(sorted(m['cfg'].items()))
        if key in seen:
            continue
        seen.add(key)
        out = os.path.join(MODELS_DIR, f'exp_{m["model_id"]}.rknn')
        if os.path.isfile(out):
            print(f'[跳过] {m["model_id"]}: {out} 已存在')
            continue
        jobs.append((m['model_id'], m['cfg'], out))

    print(f'待转换模型: {len(jobs)} 个 (并行 {args.workers})')
    if not jobs:
        print('无待转换模型')
        return

    with mp.Pool(args.workers) as pool:
        for mid, ok, err in pool.imap_unordered(convert_one, jobs):
            status = '✅' if ok else f'❌ {err}'
            print(f'  {mid}: {status}')

    print('===== 批量转换完成 =====')


if __name__ == '__main__':
    main()
