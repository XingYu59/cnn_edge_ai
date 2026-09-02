"""
Hardware Constraint Filter (Task 7)
===================================
检查 CNN 是否满足硬件约束 (latency/memory/params)。
"""
from typing import Dict, Any, List, Optional

from .latency import RKNNLatencyModel, predict_latency
from .memory import estimate_memory


def check_hardware_constraints(
        profile: Dict[str, Any],
        constraints: Dict[str, float],
        latency_model: Optional[RKNNLatencyModel] = None) -> Dict[str, Any]:
    """
    检查架构 profile 是否满足硬件约束。

    constraints 支持:
      max_latency_us   : 最大允许 latency (us)
      max_memory_bytes : 最大允许估计内存 (bytes)
      max_params       : 最大允许参数量

    返回:
      {
        feasible, latency_us, memory_bytes, params,
        violations: [描述字符串]
      }
    """
    if latency_model is None:
        latency_model = RKNNLatencyModel.load()

    latency = latency_model.predict(profile)
    mem = estimate_memory(profile)
    params = profile['params']

    violations: List[str] = []
    if 'max_latency_us' in constraints and \
            latency > constraints['max_latency_us']:
        violations.append(
            f'latency {latency:.0f}us > max {constraints["max_latency_us"]}us')
    if 'max_memory_bytes' in constraints and \
            mem['estimated_memory_bytes'] > constraints['max_memory_bytes']:
        violations.append(
            f'memory {mem["estimated_memory_bytes"]}B > '
            f'max {constraints["max_memory_bytes"]}B')
    if 'max_params' in constraints and params > constraints['max_params']:
        violations.append(
            f'params {params:,} > max {constraints["max_params"]:,}')

    return {
        'feasible': len(violations) == 0,
        'latency_us': round(float(latency), 1),
        'memory_bytes': mem['estimated_memory_bytes'],
        'params': params,
        'violations': violations,
    }
