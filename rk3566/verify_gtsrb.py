"""
GTSRB RKNN 板端验证与测速 (K11C / RK3566)
==========================================
功能:
1. 从 GTSRB 测试集取 N 张图, 在 RK3566 NPU 上推理, 统计 top1 精度
   (验证 .rknn 转换后模型是否正常工作, 与训练精度对比量化掉点)
2. 测量端到端延迟 (PC 调用) 与 NPU 纯计算时间 (eval_perf)

用法 (板子需通过 USB 连接 + rknn_server 运行):
    source /home/xing/venvs/rknn/bin/activate
    python verify_gtsrb.py --model models/cnn_d5_k3.rknn --num 50

注意: load_rknn 的模型必须在真机上推理, 不能在 PC 模拟器运行。
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'GTSRB')


def load_test_images(num: int = 50, input_size: int = 64):
    """从 GTSRB test.parquet 取 num 张图 + 真实标签 (RGB uint8 NHWC)。"""
    pq = os.path.join(DATA_ROOT, 'test.parquet')
    if not os.path.isfile(pq):
        raise FileNotFoundError(f'测试数据不存在: {pq}')
    df = pd.read_parquet(pq).iloc[:num]
    images, labels = [], []
    for _, row in df.iterrows():
        raw = row['Path']
        if isinstance(raw, dict):
            raw = raw.get('bytes')
        arr = np.frombuffer(bytes(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)      # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # RGB
        img = cv2.resize(img, (input_size, input_size))
        images.append(img)
        labels.append(int(row['ClassId']))
    return np.stack(images), np.array(labels)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/cnn_d5_k3.rknn')
    p.add_argument('--num', type=int, default=50, help='验证图片数')
    p.add_argument('--target', default='rk3566')
    p.add_argument('--bench', type=int, default=20, help='测速循环次数')
    args = p.parse_args()

    from rknn.api import RKNN

    print(f'===== GTSRB 板端验证: {args.model} =====')

    # 1. 加载模型 + 连板
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(args.model) == 0, 'load_rknn failed'
    assert rknn.init_runtime(target=args.target) == 0, \
        'init_runtime failed (请确认板子已连接、rknn_server 运行中)'

    # 2. 加载测试图
    images, labels = load_test_images(args.num)
    print(f'  测试图片: {args.num} 张 (输入 {images.shape[1:]} NHWC uint8)')

    # 3. 推理 + 精度统计
    print('--> 推理中...')
    correct = 0
    t0 = time.time()
    for i in range(len(images)):
        out = rknn.inference(inputs=[images[i:i + 1]],
                             data_format=['nhwc'])[0][0]
        prob = softmax(np.array(out))
        pred = int(np.argmax(prob))
        if pred == labels[i]:
            correct += 1
    total_time = time.time() - t0
    print(f'  Top-1 精度: {correct}/{args.num} = '
          f'{correct / args.num * 100:.1f}%  '
          f'(含 50 张推理传输耗时 {total_time:.2f}s)')

    # 4. 测速 (预热 + 循环)
    print('--> 测速中...')
    img = images[0:1]
    for _ in range(3):
        rknn.inference(inputs=[img], data_format=['nhwc'])
    times = []
    for _ in range(args.bench):
        t = time.perf_counter()
        rknn.inference(inputs=[img], data_format=['nhwc'])
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    print('=' * 46)
    print(f'  端到端平均延迟 : {times.mean():.2f} ms (min {times.min():.2f})')
    print(f'  端到端 FPS     : {1000 / times.mean():.1f}')
    print('=' * 46)

    # 5. NPU 纯计算时间 (不受传输影响)
    npu_time_ms = None
    try:
        perfs = rknn.eval_perf()
        text = str(perfs)
        for line in text.splitlines():
            if 'Total Time' in line or 'FPS' in line:
                print(f'  NPU eval_perf : {line.strip()}')
            if 'Total Time' in line:
                import re
                m = re.search(r'(\d+)', line)
                if m:
                    npu_time_ms = int(m.group(1)) / 1000.0
    except Exception as e:
        print(f'  eval_perf 不可用: {e}')

    rknn.release()
    print('=== 验证完成 ===')

    # 6. 记录到 results/board_results.csv (板端实测数据库)
    record = {
        'model': os.path.basename(args.model),
        'num_images': len(images),
        'top1_acc': round(correct / len(images), 4),
        'e2e_latency_ms': round(float(times.mean()), 4),
        'e2e_fps': round(float(1000 / times.mean()), 2),
        'npu_latency_ms': npu_time_ms,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'board_results.csv')
    import csv as _csv
    fieldnames = list(record.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    print(f'  [record] 已写入 {csv_path}')


if __name__ == '__main__':
    main()
