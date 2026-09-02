"""
Model Analyzer
==============
模型静态分析工具:
- 参数量 (total / trainable)
- MACs  (理论乘加运算量, 1 MAC = 2 FLOPs)
- 模型大小 (FP32: params x 4 bytes)
- 各主要卷积层 Feature Map (layer / input shape / output shape)

重要说明:
    MACs 是模型的理论计算量, 不是真实硬件推理时延 (latency)。
    真实时延取决于硬件算力、内存带宽、算子实现等, 需实测。
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

import numpy as np


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回 (total_params, trainable_params)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model: nn.Module, input_size: int = 64) -> int:
    """
    统计一次 forward pass 的理论 MACs (输入 1x3xinput_sizexinput_size)。

    实现: 手动遍历 Conv2d / Linear, 按公式计算, 不依赖第三方库。
      Conv2d : MACs = out_c * out_h * out_w * (in_c/groups) * k_h * k_w
      Linear : MACs = in_features * out_features
    """
    macs = 0

    def conv_macs(m: nn.Conv2d, x_shape, out_shape):
        out_c, out_h, out_w = out_shape[1], out_shape[2], out_shape[3]
        in_c = x_shape[1] // m.groups
        return out_c * out_h * out_w * in_c * m.kernel_size[0] * m.kernel_size[1]

    def linear_macs(m: nn.Linear, x_shape, out_shape):
        return x_shape[1] * m.out_features

    def hook_fn(m, x, y):
        nonlocal macs
        x = x[0]
        if isinstance(m, nn.Conv2d):
            macs += conv_macs(m, tuple(x.shape), tuple(y.shape))
        elif isinstance(m, nn.Linear):
            macs += linear_macs(m, tuple(x.shape), tuple(y.shape))

    hooks = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(torch.randn(1, 3, input_size, input_size))

    for h in hooks:
        h.remove()
    return macs


def model_size_mb(total_params: int) -> float:
    """FP32 参数理论存储量: params x 4 bytes, 输出 MB。"""
    return total_params * 4 / (1024 ** 2)


def feature_map_info(model: nn.Module, input_size: int = 64,
                     input_channels: int = 3) -> List[Dict]:
    """
    记录各主要卷积层的 Feature Map 信息:
      layer, input shape, output shape, channels, spatial resolution
    通过 forward hook 收集, 返回按前向顺序排列的列表。
    """
    records: List[Dict] = []

    def hook_fn(name):
        def _h(m, x, y):
            x = x[0]
            records.append({
                'layer': name,
                'input_shape': list(x.shape),
                'output_shape': list(y.shape),
                'channels': y.shape[1],
                'spatial': f'{y.shape[2]}x{y.shape[3]}',
            })
        return _h

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        model(torch.randn(1, input_channels, input_size, input_size))

    for h in hooks:
        h.remove()
    return records


def format_macs(macs: int) -> str:
    """把 MACs 数字格式化为可读字符串, 如 12.34M / 1.23G。"""
    if macs >= 1e9:
        return f'{macs / 1e9:.2f}G MACs ({macs:,})'
    if macs >= 1e6:
        return f'{macs / 1e6:.2f}M MACs ({macs:,})'
    return f'{macs:,} MACs'


def analyze_model(model: nn.Module, input_size: int = 64) -> Dict:
    """一站式分析, 返回统计 dict (供 results.csv 与打印使用)。"""
    total, trainable = count_parameters(model)
    macs = count_macs(model, input_size)
    return {
        'parameters': total,
        'trainable_parameters': trainable,
        'macs': macs,
        'model_size_MB': round(model_size_mb(total), 4),
    }


# ===========================================================================
# 硬件感知分析 (Task Plan 第 4/5/6 节)
# 统一 Model Description: 逐层结构 + classifier 指标 + MACs 分类
# ===========================================================================

def analyze_model_detail(model: nn.Module, input_size: int = 64,
                         input_channels: int = 3) -> Dict:
    """
    详细硬件感知分析, 返回:
      summary   : params / total_macs / conv_macs / linear_macs / other_macs
      layers    : 逐层记录 (layer_id/name/type/input_shape/output_shape/
                  kernel/stride/params/macs/macs_type)
      classifier: feature_map_shape / flatten_dim / classifier_input_dim /
                  classifier_params / classifier_macs
    全部通过 forward hook 从模型自动获取, 不手工填写。
    """
    records: List[Dict] = []
    hooks = []

    def make_hook(name: str, mod):
        def _h(m, x, y):
            x = x[0]
            in_shape = list(x.shape)
            out_shape = list(y.shape)
            params = sum(p.numel() for p in m.parameters())

            if isinstance(m, nn.Conv2d):
                macs = (out_shape[1] * out_shape[2] * out_shape[3]
                        * (in_shape[1] // m.groups)
                        * m.kernel_size[0] * m.kernel_size[1])
                macs_type = 'conv'
                kernel = list(m.kernel_size)
                stride = list(m.stride)
            elif isinstance(m, nn.Linear):
                macs = in_shape[1] * m.out_features
                macs_type = 'linear'
                kernel, stride = None, None
            elif isinstance(m, nn.MaxPool2d):
                macs = 0
                macs_type = 'pool'
                kernel = list(m.kernel_size) if isinstance(m.kernel_size, tuple) else [m.kernel_size]
                stride = list(m.stride) if isinstance(m.stride, tuple) else [m.stride]
            else:
                macs = 0
                macs_type = 'other'
                kernel, stride = None, None

            records.append({
                'name': name,
                'type': type(m).__name__,
                'input_shape': in_shape,
                'output_shape': out_shape,
                'kernel': kernel,
                'stride': stride,
                'params': params,
                'macs': macs,
                'macs_type': macs_type,
            })
        return _h

    # 注册 hook (按前向执行顺序触发)
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.MaxPool2d,
                            nn.BatchNorm2d, nn.ReLU)):
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    with torch.no_grad():
        model(torch.randn(1, input_channels, input_size, input_size))
    for h in hooks:
        h.remove()

    # 补充 layer_id (按执行顺序编号)
    for i, r in enumerate(records, 1):
        r['layer_id'] = i

    # ---- 汇总 MACs 分类 ----
    conv_macs = sum(r['macs'] for r in records if r['macs_type'] == 'conv')
    linear_macs = sum(r['macs'] for r in records if r['macs_type'] == 'linear')
    other_macs = sum(r['macs'] for r in records
                     if r['macs_type'] not in ('conv', 'linear'))
    total_params = sum(p.numel() for p in model.parameters())

    # ---- classifier 指标 (Linear 层即分类头) ----
    linear_records = [r for r in records if r['macs_type'] == 'linear']
    if linear_records:
        # feature map = 最后一个非 linear 层的输出 (flatten 前)
        non_linear = [r for r in records if r['macs_type'] != 'linear']
        fm_shape = non_linear[-1]['output_shape'][1:] if non_linear else None
        flatten_dim = linear_records[0]['input_shape'][1]
        classifier_params = sum(r['params'] for r in linear_records)
        classifier_macs = sum(r['macs'] for r in linear_records)
    else:
        fm_shape = None
        flatten_dim = 0
        classifier_params = 0
        classifier_macs = 0

    classifier = {
        'feature_map_shape': fm_shape,          # [C, H, W]
        'feature_map_channels': fm_shape[0] if fm_shape else 0,
        'feature_map_height': fm_shape[1] if fm_shape else 0,
        'feature_map_width': fm_shape[2] if fm_shape else 0,
        'flatten_dimension': flatten_dim,
        'classifier_input_dimension': flatten_dim,   # = C×H×W
        'classifier_parameters': classifier_params,
        'classifier_macs': classifier_macs,
    }

    summary = {
        'parameters': total_params,
        'total_macs': conv_macs + linear_macs + other_macs,
        'conv_macs': conv_macs,
        'linear_macs': linear_macs,
        'other_macs': other_macs,
        'model_size_MB': round(model_size_mb(total_params), 4),
    }

    return {'summary': summary, 'layers': records,
            'classifier': classifier}


def print_model_detail(detail: Dict, model_name: str = '') -> None:
    """友好打印 analyze_model_detail 的结果。"""
    s, c = detail['summary'], detail['classifier']
    print(f'===== {model_name} 硬件感知分析 =====')
    print(f'  Parameters: {s["parameters"]:,} | '
          f'Model Size: {s["model_size_MB"]:.2f} MB')
    print(f'  MACs 分类: total={s["total_macs"]:,} | '
          f'conv={s["conv_macs"]:,} | linear={s["linear_macs"]:,} | '
          f'other={s["other_macs"]:,}')
    if c['feature_map_shape']:
        print(f'  Feature Map: {c["feature_map_shape"][0]}×'
              f'{c["feature_map_shape"][1]}×{c["feature_map_shape"][2]} '
              f'| Flatten: {c["flatten_dimension"]}')
        print(f'  Classifier: params={c["classifier_parameters"]:,} | '
              f'MACs={c["classifier_macs"]:,} | '
              f'输入维度={c["classifier_input_dimension"]}')
    print(f'  {"ID":<3} {"Type":<12} {"InShape":<18} {"OutShape":<18} '
          f'{"Kernel":<8} {"MACs":<10} {"Type"}')
    for r in detail['layers']:
        print(f'  {r["layer_id"]:<3} {r["type"]:<12} '
              f'{str(r["input_shape"]):<18} {str(r["output_shape"]):<18} '
              f'{str(r["kernel"]):<8} {r["macs"]:<10,} {r["macs_type"]}')
