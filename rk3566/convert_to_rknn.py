"""
Convert GTSRB CNN checkpoint (.pt) to RKNN model (.rknn)
=========================================================
参考: rknn-toolkit2-2.3.2/examples/pytorch/resnet18/test.py

流程:
    checkpoint + 模型结构参数
      → build_model 重建网络 + 加载训练权重
      → torch.jit.trace 导出 TorchScript (.pt)
      → rknn.load_pytorch → build(int8 量化) → export_rknn
      → (可选) PC 模拟器推理验证 Top-5

用法 (在 cnn_rk3566 目录下, 使用 rknn 虚拟环境):
    source /home/xing/venvs/rknn/bin/activate
    python convert_to_rknn.py \
        --ckpt results/cnn_d5_c32-32-64-64-128_k3_best.pt \
        --depth 5 --channels 32,32,64,64,128 --kernel-size 3 \
        --out models/cnn_d5_k3.rknn

关键点:
1. mean/std 换算: torch 训练在 0-1 域归一化, RKNN 在 0-255 域,
   所以 rknn.config 的 mean/std = 训练值 × 255。
2. 量化校准集: 从 GTSRB parquet 抽取 N 张图导出为文件, 供 rknn.build 校准。
3. 本脚本不依赖 GPU 和训练环境, 用 rknn-toolkit2 环境即可运行。
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2

# 允许从项目根目录 import modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.generator import build_model, validate_model_config
from modules.dataset import GTSRB_MEAN, GTSRB_STD

DATA_ROOT = os.path.join(BASE_DIR, 'data', 'GTSRB')
CALIB_DIR = os.path.join(BASE_DIR, 'data', 'calib')
DATASET_TXT = os.path.join(BASE_DIR, 'data', 'calib', 'dataset.txt')


def parse_args():
    p = argparse.ArgumentParser(description='GTSRB CNN .pt -> .rknn 转换')
    p.add_argument('--ckpt', default=None,
                   help='训练好的 checkpoint (.pt); 不提供则用随机权重 (仅 latency 实验)')
    p.add_argument('--out', default=None, help='输出 .rknn 路径')
    p.add_argument('--depth', type=int, required=True, help='卷积层数')
    p.add_argument('--channels', required=True, help='通道数, 逗号分隔, 如 32,32,64,64,128')
    p.add_argument('--kernel-size', type=int, default=3, help='卷积核 (3/5)')
    p.add_argument('--input-size', type=int, default=64, help='输入尺寸')
    p.add_argument('--num-classes', type=int, default=43, help='分类数')
    p.add_argument('--target', default='rk3566', help='目标平台')
    p.add_argument('--calib-num', type=int, default=100, help='量化校准图片数')
    p.add_argument('--verify', action='store_true', help='转换后模拟器验证 Top-5')
    return p.parse_args()


def export_calib_images(num: int) -> str:
    """从 GTSRB parquet 导出校准图片文件, 生成 dataset.txt, 返回其路径。"""
    import pandas as pd
    os.makedirs(CALIB_DIR, exist_ok=True)
    pq = os.path.join(DATA_ROOT, 'train.parquet')
    if not os.path.isfile(pq):
        raise FileNotFoundError(f'校准数据源不存在: {pq}')
    df = pd.read_parquet(pq)
    # 均匀采样, 覆盖各类别
    df = (df.groupby('ClassId', group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, num // 43)),
                                      random_state=42))
            .reset_index(drop=True))
    lines = []
    for i, row in df.iterrows():
        raw = row['Path']
        if isinstance(raw, dict):
            raw = raw.get('bytes')
        img_path = os.path.join(CALIB_DIR, f'calib_{i:05d}.png')
        with open(img_path, 'wb') as f:
            f.write(bytes(raw))
        lines.append(img_path)
    with open(DATASET_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  [calib] 导出 {len(lines)} 张校准图 -> {DATASET_TXT}')
    return DATASET_TXT


def export_torchscript(model, input_size: int, pt_path: str):
    """重建模型 + 加载权重 + trace 导出 TorchScript .pt (参考 resnet18 示例)。"""
    import torch
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(pt_path)
    print(f'  [export] TorchScript 已保存: {pt_path}')


def convert_to_rknn(ckpt_path, model_cfg, out_path, target, calib_num,
                    verify=False):
    import torch
    from rknn.api import RKNN

    # ---------- 1. 重建模型 + 加载训练权重 (可选) ----------
    print('===== 1. 重建模型 =====')
    model = build_model(model_cfg)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  加载 checkpoint: {ckpt_path} | '
              f'best_val_acc={ckpt.get("best_val_acc", "N/A")}')
    else:
        # 随机权重: 仅用于 latency 测量 (latency 只取决于结构, 与权重值无关)
        print(f'  [随机权重] 未加载 checkpoint (仅 latency 实验用)')
    model.eval()

    # ---------- 2. 导出 TorchScript ----------
    print('===== 2. 导出 TorchScript (.pt) =====')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    pt_path = out_path.replace('.rknn', '.pt')
    export_torchscript(model, model_cfg['input_size'], pt_path)

    # ---------- 3. 导出量化校准图 ----------
    print('===== 3. 导出量化校准图 =====')
    dataset_txt = export_calib_images(calib_num)

    # ---------- 4. RKNN 转换 (参考 test.py) ----------
    print('===== 4. RKNN 转换 =====')
    rknn = RKNN(verbose=True)

    # mean/std 换算: torch 0-1 域 × 255 -> RKNN 0-255 域
    mean = [m * 255 for m in GTSRB_MEAN]
    std = [s * 255 for s in GTSRB_STD]
    print(f'  mean_values={mean}, std_values={std}, target={target}')

    print('--> Config model')
    ret = rknn.config(mean_values=mean, std_values=std,
                      target_platform=target)
    assert ret == 0, 'config failed'

    print('--> Loading model')
    ret = rknn.load_pytorch(model=pt_path,
                            input_size_list=[[1, 3, model_cfg['input_size'],
                                              model_cfg['input_size']]])
    assert ret == 0, 'load_pytorch failed'

    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    assert ret == 0, 'build failed'

    print('--> Export rknn model')
    ret = rknn.export_rknn(out_path)
    assert ret == 0, 'export failed'

    # ---------- 5. (可选) 模拟器验证 ----------
    if verify:
        print('===== 5. 模拟器验证 Top-5 =====')
        ret = rknn.init_runtime()
        assert ret == 0, 'init_runtime failed (模拟器)'
        img = load_calib_image()
        outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
        show_top5(outputs[0][0])

    rknn.release()
    print(f'\n===== 完成: {out_path} =====')
    print(f'  模型大小: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB')


def load_calib_image():
    """加载一张校准图作为验证输入 (uint8, NHWC)。"""
    import pandas as pd
    pq = os.path.join(DATA_ROOT, 'train.parquet')
    df = pd.read_parquet(pq)
    raw = df['Path'].iloc[0]
    if isinstance(raw, dict):
        raw = raw.get('bytes')
    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # RGB
    img = cv2.resize(img, (64, 64))
    return np.expand_dims(img, 0)


def show_top5(output):
    labels_path = os.path.join(BASE_DIR, 'data', 'gtsrb_labels.txt')
    labels = []
    if os.path.isfile(labels_path):
        labels = [l.strip() for l in open(labels_path)]
    else:
        labels = [str(i) for i in range(43)]
    prob = np.exp(output) / np.sum(np.exp(output))
    idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
    print('  ----- TOP 5 -----')
    for i in range(5):
        print(f'  [{idx[i]:>2d}] score:{prob[idx[i]]:.4f} '
              f'class:"{labels[idx[i]]}"')


def main():
    args = parse_args()
    channels = [int(c) for c in args.channels.split(',')]
    model_cfg = {
        'name': (os.path.basename(args.ckpt).replace('_best.pt', '')
                 if args.ckpt else 'random_weights'),
        'depth': args.depth,
        'channels': channels,
        'kernel_size': args.kernel_size,
        'num_classes': args.num_classes,
        'input_size': args.input_size,
    }
    validate_model_config(model_cfg)

    out_path = args.out or os.path.join(
        BASE_DIR, 'models', f'{model_cfg["name"]}.rknn')
    if args.ckpt and not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'checkpoint 不存在: {args.ckpt}')

    t0 = time.time()
    convert_to_rknn(args.ckpt, model_cfg, out_path, args.target,
                    args.calib_num, args.verify)
    print(f'  总耗时: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
