"""
Controlled Experiments (Task Plan 第二阶段)
=============================================
4 组控制变量实验, 每组只改变一个自变量:

  A_depth      : depth 3~7 递增, 保持 flatten=16384, kernel=3
  B_kernel     : 固定拓扑, 全k3 / 全k5 / 混合 (每层 kernel)
  C_featuremap : 固定 channels, 不同 pool_positions → 不同最终 FM/flatten
  D_classifier : 不同 flatten 维度 (8192/16384/32768/65536) → GEMM 规模

控制变量原则 (Task 第 9 节):
  每组只变一个自变量, 其余 (input_size/kernel 或 depth/pool/classifier) 尽量不变。
  已有 4 个模型 (d3_k3/cnn_test/d5_k3/d5_k5) 作为 legacy 复用。

注意: 本阶段实验只测 latency, 新模型使用随机权重 (latency 与权重值无关),
      不训练 (省 GPU 时间)。已有模型使用训练好的 checkpoint。
"""
from typing import Dict, List

# 所有模型共用的基础配置
_BASE = {'num_classes': 43, 'input_size': 64}

# ---------------------------------------------------------------------------
# 实验组 A: Depth (第 5 节)
# 自变量: depth 3~7 | 控制: kernel=3, 最终 flatten=16384, 首层/末层通道
# ---------------------------------------------------------------------------
GROUP_A = [
    {'model_id': 'A1', 'var': 'depth=3', 'cfg': dict(
        _BASE, depth=3, channels=[16, 32, 64], kernel_size=3,
        pool_positions=[2, 3])},
    {'model_id': 'A2', 'var': 'depth=4', 'cfg': dict(
        _BASE, depth=4, channels=[16, 32, 32, 64], kernel_size=3,
        pool_positions=[2, 4])},
    {'model_id': 'A3', 'var': 'depth=5', 'cfg': dict(
        _BASE, depth=5, channels=[16, 32, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 5])},
    {'model_id': 'A4', 'var': 'depth=6', 'cfg': dict(
        _BASE, depth=6, channels=[16, 32, 32, 64, 64, 64], kernel_size=3,
        pool_positions=[2, 6])},
    {'model_id': 'A5', 'var': 'depth=7', 'cfg': dict(
        _BASE, depth=7, channels=[16, 32, 32, 64, 64, 64, 64], kernel_size=3,
        pool_positions=[2, 7])},
]

# ---------------------------------------------------------------------------
# 实验组 B: Kernel (第 6 节)
# 自变量: kernel 配置 | 控制: depth=4, channels=[32,32,64,64], pool=[2,4]
# ---------------------------------------------------------------------------
GROUP_B = [
    {'model_id': 'B1', 'var': 'k=[3,3,3,3]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=[3, 3, 3, 3],
        pool_positions=[2, 4])},
    {'model_id': 'B2', 'var': 'k=[5,5,5,5]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=[5, 5, 5, 5],
        pool_positions=[2, 4])},
    {'model_id': 'B3', 'var': 'k=[3,3,5,5]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=[3, 3, 5, 5],
        pool_positions=[2, 4])},
    {'model_id': 'B4', 'var': 'k=[5,5,3,3]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=[5, 5, 3, 3],
        pool_positions=[2, 4])},
]

# ---------------------------------------------------------------------------
# 实验组 C: Feature Map (第 7 节, 最重要)
# 自变量: pool_positions → 最终 FM 尺寸 | 控制: depth=4, channels=[32,32,64,64], k=3
# ---------------------------------------------------------------------------
GROUP_C = [
    {'model_id': 'C1', 'var': 'FM=32x32x64', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[4])},
    {'model_id': 'C2', 'var': 'FM=16x16x64', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 4])},
    {'model_id': 'C3', 'var': 'FM=8x8x64', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 3, 4])},
    {'model_id': 'C4', 'var': 'FM=16x16x64,pool=[3,4]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[3, 4])},
    {'model_id': 'C5', 'var': 'FM=16x16x64,pool=[2,3]', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 3])},
]

# ---------------------------------------------------------------------------
# 实验组 D: Classifier / GEMM (第 8 节, 第二个重点)
# 自变量: flatten 维度 | 通过 FM 组合实现 (C×H×W)
# ---------------------------------------------------------------------------
GROUP_D = [
    {'model_id': 'D1', 'var': 'flatten=8192', 'cfg': dict(
        _BASE, depth=4, channels=[32, 64, 64, 128], kernel_size=3,
        pool_positions=[2, 3, 4])},
    {'model_id': 'D2', 'var': 'flatten=16384', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 4]),
     'rknn': 'models/exp_B1.rknn'},   # 与 B1 同配置, 复用
    {'model_id': 'D3', 'var': 'flatten=32768', 'cfg': dict(
        _BASE, depth=4, channels=[32, 64, 64, 128], kernel_size=3,
        pool_positions=[2, 4])},
    {'model_id': 'D4', 'var': 'flatten=65536', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 64], kernel_size=3,
        pool_positions=[4]),
     'rknn': 'models/exp_C1.rknn'},   # 与 C1 同配置, 复用
]

# ---------------------------------------------------------------------------
# Legacy 模型 (已有, 训练过, 复用已有 rknn)
# ---------------------------------------------------------------------------
LEGACY = [
    {'model_id': 'd3_k3', 'var': 'legacy', 'cfg': dict(
        _BASE, depth=3, channels=[16, 32, 32], kernel_size=3),
     'trained': True, 'rknn': 'models/cnn_d3_k3.rknn'},
    {'model_id': 'cnn_test', 'var': 'legacy', 'cfg': dict(
        _BASE, depth=4, channels=[16, 32, 64, 64], kernel_size=3),
     'trained': True, 'rknn': 'models/cnn_test.rknn'},
    {'model_id': 'd5_k3', 'var': 'legacy', 'cfg': dict(
        _BASE, depth=5, channels=[32, 32, 64, 64, 128], kernel_size=3),
     'trained': True, 'rknn': 'models/cnn_d5_k3.rknn'},
    {'model_id': 'd5_k5', 'var': 'legacy', 'cfg': dict(
        _BASE, depth=5, channels=[32, 32, 64, 64, 128], kernel_size=5),
     'trained': True, 'rknn': 'models/cnn_d5_k5.rknn'},
]


def _tag(group: str, items: List[Dict]) -> List[Dict]:
    """给组内模型打组标签 + rknn 路径 + trained 标记。"""
    out = []
    for it in items:
        mid = it['model_id']
        out.append({
            'model_id': mid,
            'group': group,
            'independent_var': it['var'],
            'cfg': it['cfg'],
            'trained': it.get('trained', False),
            'rknn': it.get('rknn', f'models/exp_{mid}.rknn'),
        })
    return out


ALL_GROUPS = {
    'A_depth': GROUP_A,
    'B_kernel': GROUP_B,
    'C_featuremap': GROUP_C,
    'D_classifier': GROUP_D,
}

ALL_MODELS = (_tag('A_depth', GROUP_A) + _tag('B_kernel', GROUP_B)
              + _tag('C_featuremap', GROUP_C) + _tag('D_classifier', GROUP_D)
              + _tag('legacy', LEGACY))


def get_models(group: str = None) -> List[Dict]:
    if group is None:
        return ALL_MODELS
    return [m for m in ALL_MODELS if m['group'] == group]
