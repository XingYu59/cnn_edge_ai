"""
Minimal Pipeline (Task 8/10)
============================
CNN Config -> Generator -> Analyzer -> Latency -> Memory -> Filter

evaluate_candidate 是下一阶段 Hardware-aware Search 的接口:
  for arch in generator():
      r = evaluate_candidate(arch)
      if r['feasible']: candidates.append(r)
"""
from typing import Dict, Any, Optional

from .architecture import analyze_architecture
from .latency import RKNNLatencyModel, predict_latency
from .memory import (memory_profile, estimate_memory,
                     RKNNMemoryModel, predict_runtime_memory)
from .filter import check_hardware_constraints


def evaluate_candidate(config: Dict[str, Any],
                       constraints: Optional[Dict[str, float]] = None,
                       latency_model: Optional[RKNNLatencyModel] = None,
                       memory_model: Optional[RKNNMemoryModel] = None
                       ) -> Dict[str, Any]:
    """
    对单个 CNN 配置做完整硬件评估。

    返回:
      {
        architecture, params, total_macs, conv_macs, linear_macs,
        flatten_dim, predicted_latency_us,
        weight_memory_bytes, peak_activation_memory_bytes,
        estimated_peak_memory_bytes, predicted_runtime_memory_bytes,
        feasible, violations
      }
    """
    profile = analyze_architecture(config)

    if latency_model is None:
        latency_model = RKNNLatencyModel.load()
    latency = latency_model.predict(profile)
    mem = memory_profile(profile, weight_precision='int8',
                         activation_precision='int8')
    if memory_model is None:
        memory_model = RKNNMemoryModel.load()
    runtime_mem = memory_model.predict_runtime_memory(profile)

    if constraints:
        check = check_hardware_constraints(profile, constraints,
                                           latency_model)
        feasible, violations = check['feasible'], check['violations']
    else:
        feasible, violations = True, []

    return {
        'architecture': config,
        'params': profile['params'],
        'total_macs': profile['total_macs'],
        'conv_macs': profile['conv_macs'],
        'linear_macs': profile['linear_macs'],
        'flatten_dim': profile['flatten_dim'],
        'predicted_latency_us': round(float(latency), 1),
        'weight_memory_bytes': mem['weight_memory_bytes'],
        'peak_activation_memory_bytes': mem['peak_activation_memory_bytes'],
        'estimated_peak_memory_bytes': mem['estimated_peak_memory_bytes'],
        'predicted_runtime_memory_bytes': runtime_mem,
        'feasible': feasible,
        'violations': violations,
    }
