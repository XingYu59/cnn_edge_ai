"""
Memory Estimator & Predictor (Task Plan: Memory Analysis)
=========================================================
第一版 RK3566 CNN Memory Model (纯静态估算, 非 Runtime 实测):

  M_weight       = params × bytes_per_param            (INT8=1 / FP16=2 / FP32=4)
  M_activation_i = C_i × H_i × W_i × bytes            (每层)
  M_peak_act     = max(M_activation_i)                 (最大单层近似)
  M_estimated    = M_weight + M_peak_act

注意 (Phase 2):
  - 模型文件大小 ≠ Runtime Memory
  - estimated ≠ measured; 未校准前不声称等于 RKNN Runtime 真实内存
  - 实际 Runtime 峰值可能更高 (input/output buffers, workspace, alignment...)
"""
from typing import Dict, Any, List, Optional

PRECISION_BYTES = {'int8': 1, 'fp16': 2, 'fp32': 4}


def memory_profile(profile: Dict[str, Any],
                   weight_precision: str = 'int8',
                   activation_precision: str = 'int8') -> Dict[str, Any]:
    """
    完整 Memory Profile (Phase 3/4)。

    输入: analyze_architecture() 的 profile
    输出: {
      params, weight_memory_bytes,
      peak_activation_memory_bytes, estimated_peak_memory_bytes,
      max_feature_map, max_feature_map_shape,
      largest_activation_layer, largest_activation_layer_type,
      activation_layers: [{layer, shape, elements, memory_bytes}]
      precision: {weight, activation}
    }
    """
    if weight_precision not in PRECISION_BYTES:
        raise ValueError(f'不支持的权重精度: {weight_precision}')
    if activation_precision not in PRECISION_BYTES:
        raise ValueError(f'不支持的激活精度: {activation_precision}')

    weight_bytes = profile['params'] * PRECISION_BYTES[weight_precision]

    # 每层 activation (feature_maps: 每层 [C,H,W])
    act_bytes = PRECISION_BYTES[activation_precision]
    activation_layers: List[Dict] = []
    for i, (c, h, w) in enumerate(profile['feature_maps']):
        elements = c * h * w
        activation_layers.append({
            'layer': f'conv_{i+1}',
            'shape': [c, h, w],
            'elements': elements,
            'memory_bytes': elements * act_bytes,
        })

    if activation_layers:
        largest = max(activation_layers, key=lambda x: x['memory_bytes'])
        peak_act = largest['memory_bytes']
        largest_layer = largest['layer']
        largest_type = 'Conv2d'
        max_fm = largest['elements']
        max_fm_shape = largest['shape']
    else:
        peak_act = 0
        largest_layer = None
        largest_type = None
        max_fm = 0
        max_fm_shape = None

    return {
        'params': profile['params'],
        'weight_memory_bytes': weight_bytes,
        'peak_activation_memory_bytes': peak_act,
        'estimated_peak_memory_bytes': weight_bytes + peak_act,
        'max_feature_map': max_fm,
        'max_feature_map_shape': max_fm_shape,
        'largest_activation_layer': largest_layer,
        'largest_activation_layer_type': largest_type,
        'activation_layers': activation_layers,
        'precision': {
            'weight': weight_precision,
            'activation': activation_precision,
        },
        'note': 'estimated (静态估算, 未经过 RKNN Runtime calibration)',
    }


def estimate_memory(profile: Dict[str, Any],
                    weight_precision: str = 'int8',
                    activation_bytes: int = 1) -> Dict[str, Any]:
    """兼容旧接口: 简化版估算 (int8 激活按给定 bytes)。"""
    wb = profile['params'] * PRECISION_BYTES[weight_precision]
    act = profile['max_feature_map_size'] * activation_bytes
    return {
        'weight_memory_bytes': wb,
        'max_activation_memory_bytes': act,
        'estimated_memory_bytes': wb + act,
        'weight_precision': weight_precision,
        'note': 'estimated (静态估算, 非 RKNN Runtime 实测)',
    }


def predict_memory(profile: Dict[str, Any],
                   weight_precision: str = 'int8',
                   activation_precision: str = 'int8') -> Dict[str, Any]:
    """
    Memory Predictor 接口 (Phase 10): 与 predict_latency 对齐。

    输入: architecture profile
    输出: weight_memory / peak_activation_memory / estimated_peak_memory
    """
    return memory_profile(profile, weight_precision, activation_precision)


# ---------------------------------------------------------------------------
# 硬件校准模型 (Phase 9 情况 B, 基于 10 模型实测)
# ---------------------------------------------------------------------------
import json as _json
import os as _os

DEFAULT_MEMORY_MODEL_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'models', 'rknn_memory_v1.json')


class RKNNMemoryModel:
    """RK3566 Runtime Memory 校准模型 (分段线性)。"""

    def __init__(self, data: Dict):
        self.data = data
        self.threshold = data['flatten_threshold']
        self.cal = data['calibration']

    @classmethod
    def load(cls, path: str = DEFAULT_MEMORY_MODEL_PATH) -> 'RKNNMemoryModel':
        with open(path) as f:
            return cls(_json.load(f))

    def predict_runtime_memory(self, profile: Dict,
                               weight_precision: str = 'int8',
                               activation_precision: str = 'int8') -> int:
        """
        估算 + 校准 → 预测 Runtime Memory (bytes)。

        先做静态估算 (estimated_peak), 再按 flatten 阈值选校准组修正。
        """
        est = memory_profile(profile, weight_precision,
                             activation_precision)['estimated_peak_memory_bytes']
        group = ('int8_path' if profile['flatten_dim'] <= self.threshold
                 else 'fp16gemm_path')
        cal = self.cal[group]
        return int(cal['a'] * est + cal['b'])


def predict_runtime_memory(profile: Dict,
                           model: Optional['RKNNMemoryModel'] = None) -> int:
    """便捷函数: 预测 RK3566 Runtime Memory (bytes)。"""
    if model is None:
        model = RKNNMemoryModel.load()
    return model.predict_runtime_memory(profile)
