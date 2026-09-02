"""
GTSRB CNN → NCNN (param/bin) 转换 (H618 阶段)
=============================================
链路: build_model → ONNX → pnnx → ncnn param/bin

模型使用随机权重 (latency 只依赖网络结构, 与权重值无关),
与 RK3566 控制变量实验同一做法。输出到 models/。

用法: python convert_to_ncnn.py [model_id...]   (不传则转全部)
"""
import os
import subprocess
import sys

import torch

# rk3566 项目路径 (复用 Generator)
RK3566_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'rk3566')
sys.path.insert(0, RK3566_DIR)

from modules.generator import build_model

PNNX = '/home/xing/venvs/rknn/bin/pnnx'
H618_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(H618_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

_BASE = {'num_classes': 43, 'input_size': 64}

# 首批: RK3566 已有实测的 legacy 模型 (保证双平台同结构)
MODELS = {
    'cnn_test': dict(_BASE, depth=4, channels=[16, 32, 64, 64],
                     kernel_size=3),
    'd3_k3': dict(_BASE, depth=3, channels=[16, 32, 32], kernel_size=3),
    'd5_k3': dict(_BASE, depth=5, channels=[32, 32, 64, 64, 128],
                  kernel_size=3),
    'd5_k5': dict(_BASE, depth=5, channels=[32, 32, 64, 64, 128],
                  kernel_size=5),
}

# 更多模型: 从 rk3566 项目自动获取 (控制变量实验 + 验证集, 均有 RK3566 实测)
EXTRA_POOL = {}


def load_extra_models():
    """从 rk3566/controlled_experiments + validation_models 载入模型配置。"""
    from controlled_experiments import ALL_MODELS as EXP
    from validation_models import VALIDATION_MODELS
    for m in EXP:
        if m['group'] != 'legacy':
            EXTRA_POOL[m['model_id']] = dict(_BASE, **m['cfg'])
    for m in VALIDATION_MODELS:
        EXTRA_POOL[m['model_id']] = dict(_BASE, **m['cfg'])


def convert_one(model_id: str, cfg: dict):
    print(f'===== {model_id} =====')
    # 1. 构建 (随机权重)
    model = build_model(cfg)
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)

    # 2. ONNX
    onnx_path = os.path.join(MODELS_DIR, f'{model_id}.onnx')
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11)
    print(f'  ONNX: {onnx_path}')

    # 3. pnnx → ncnn
    ncnn_param = os.path.join(MODELS_DIR, f'{model_id}.param')
    ncnn_bin = os.path.join(MODELS_DIR, f'{model_id}.bin')
    ret = subprocess.run(
        [PNNX, onnx_path,
         f'ncnnparam={ncnn_param}', f'ncnnbin={ncnn_bin}',
         'inputshape=[1,3,64,64]', 'optlevel=2'],
        capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'  pnnx 失败: {ret.stderr[-500:]}')
        return False
    # 清理 pnnx 中间产物
    for f in os.listdir(MODELS_DIR):
        if model_id in f and not f.endswith(('.param', '.bin', '.onnx')):
            os.remove(os.path.join(MODELS_DIR, f))
    print(f'  NCNN: {ncnn_param} + {ncnn_bin}'
          f' ({os.path.getsize(ncnn_bin)//1024}KB)')
    return True


def main():
    load_extra_models()
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for mid in targets:
        cfg = MODELS.get(mid) or EXTRA_POOL.get(mid)
        if cfg is None:
            print(f'未知模型: {mid}')
            continue
        if os.path.isfile(os.path.join(MODELS_DIR, f'{mid}.bin')):
            print(f'[跳过] {mid} 已存在')
            continue
        convert_one(mid, cfg)
    print('===== 完成 =====')


if __name__ == '__main__':
    main()
