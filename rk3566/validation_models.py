"""
Independent Validation Models (Task Plan 第三阶段, 第 4 节)
============================================================
12 个新 CNN, 只用于验证 M1/M2 的泛化能力。

★ 这些模型绝不参与 M1/M2 拟合。

覆盖范围:
  depth    : 3~6
  kernel   : 3×3 / 5×5 / 混合
  FM       : 8×8 / 16×16 / 32×32
  flatten  : 2048 / 8192 / 16384 / 32768 / 65536 (classifier small~large)

结构与训练集 (A/B/C/D 组) 的组合不同 (不同首层宽度/通道组合/pool 位置)。
"""
_BASE = {'num_classes': 43, 'input_size': 64}

VALIDATION_MODELS = [
    {'model_id': 'V1', 'cfg': dict(
        _BASE, depth=3, channels=[32, 16, 64], kernel_size=3,
        pool_positions=[2, 3])},
    {'model_id': 'V2', 'cfg': dict(
        _BASE, depth=4, channels=[64, 32, 64, 64], kernel_size=3,
        pool_positions=[2, 4])},
    {'model_id': 'V3', 'cfg': dict(
        _BASE, depth=5, channels=[16, 32, 64, 64, 32], kernel_size=3,
        pool_positions=[2, 4, 5])},
    {'model_id': 'V4', 'cfg': dict(
        _BASE, depth=4, channels=[32, 32, 64, 128], kernel_size=5,
        pool_positions=[2, 4])},
    {'model_id': 'V5', 'cfg': dict(
        _BASE, depth=3, channels=[64, 64, 128], kernel_size=3,
        pool_positions=[2, 3])},
    {'model_id': 'V6', 'cfg': dict(
        _BASE, depth=5, channels=[32, 32, 64, 128, 128], kernel_size=3,
        pool_positions=[3, 5])},
    {'model_id': 'V7', 'cfg': dict(
        _BASE, depth=4, channels=[16, 32, 64, 128], kernel_size=3,
        pool_positions=[2, 4])},
    {'model_id': 'V8', 'cfg': dict(
        _BASE, depth=4, channels=[32, 64, 64, 128], kernel_size=3,
        pool_positions=[2, 3])},
    {'model_id': 'V9', 'cfg': dict(
        _BASE, depth=6, channels=[16, 32, 32, 64, 64, 128], kernel_size=3,
        pool_positions=[2, 4, 6])},
    {'model_id': 'V10', 'cfg': dict(
        _BASE, depth=5, channels=[32, 32, 64, 64, 128], kernel_size=5,
        pool_positions=[2, 4, 5])},
    {'model_id': 'V11', 'cfg': dict(
        _BASE, depth=3, channels=[32, 32, 64], kernel_size=3,
        pool_positions=[3])},
    {'model_id': 'V12', 'cfg': dict(
        _BASE, depth=4, channels=[64, 64, 64, 128], kernel_size=3,
        pool_positions=[2, 4])},
]
