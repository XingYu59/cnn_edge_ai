"""
Architecture Analyzer (Task 2)
==============================
给定 CNN 配置, 纯静态分析得到硬件特征 profile (不依赖 RK3566 实机)。

复用现有 modules/generator + modules/analyzer, 不重复实现。
"""
import sys
import os
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.generator import build_model, validate_model_config
from modules.analyzer import analyze_model_detail


def analyze_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    静态分析 CNN 配置, 返回硬件特征 profile。

    输入: 模型配置 (depth/channels/kernel_size/pool_positions/...)
    输出: {
        depth, channels, kernel_size, params, total_macs, conv_macs,
        linear_macs, flatten_dim, feature_maps, max_feature_map_size
    }
    """
    config = validate_model_config(config)
    detail = analyze_model_detail(build_model(config))
    s, c = detail['summary'], detail['classifier']

    # 各卷积层输出特征图 (C×H×W)
    conv_out = [r['output_shape'][1:] for r in detail['layers']
                if r['macs_type'] == 'conv']
    max_fm = max((c_ * h * w for c_, h, w in conv_out), default=0)

    return {
        'depth': config['depth'],
        'channels': config['channels'],
        'kernel_size': config['kernel_size'],
        'params': s['parameters'],
        'total_macs': s['total_macs'],
        'conv_macs': s['conv_macs'],
        'linear_macs': s['linear_macs'],
        'flatten_dim': c['flatten_dimension'],
        'feature_maps': conv_out,               # 每层 [C,H,W]
        'max_feature_map_size': max_fm,         # 最大单层元素数
    }


def profile_to_row(profile: Dict[str, Any]) -> Dict[str, Any]:
    """把 profile 转成可写入 CSV 的扁平行 (Task 2 统一数据表)。"""
    return {
        'model_id': 'arch',
        'depth': profile['depth'],
        'channels': str(profile['channels']),
        'kernel_size': str(profile['kernel_size']),
        'params': profile['params'],
        'total_macs': profile['total_macs'],
        'conv_macs': profile['conv_macs'],
        'linear_macs': profile['linear_macs'],
        'flatten_dim': profile['flatten_dim'],
        'max_feature_map_size': profile['max_feature_map_size'],
    }
