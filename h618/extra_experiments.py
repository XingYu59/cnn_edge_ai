"""
扩展实验模型清单 (Phase 2-7)
============================
新增模型 (~11 个), 重点是:
  D 组: Flatten 分级 (4096~65536), conv backbone 接近 —— 验证 Flatten 双平台效应
  XL 组: 超大 MACs (验证 MACs scaling 边界)
  均匀组: MACs 中间采样 (让 predictor 数据分布更均匀)

输出: EXTRA_MODELS (model_id -> {group, cfg}), 供转换脚本使用。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'rk3566'))

from modules.generator import build_model, validate_model_config
from modules.analyzer import analyze_model_detail

_BASE = {'num_classes': 43, 'input_size': 64}

# D 组: Flatten 分级 (4K/8K/16K/32K/64K), backbone [32,64,64,X]
GROUP_D = {
    'FD4K': dict(_BASE, depth=4, channels=[32, 64, 64, 64],
                 kernel_size=3, pool_positions=[2, 3, 4]),
    'FD8K': dict(_BASE, depth=4, channels=[32, 64, 64, 128],
                 kernel_size=3, pool_positions=[2, 3, 4]),
    'FD16K': dict(_BASE, depth=4, channels=[32, 64, 64, 64],
                  kernel_size=3, pool_positions=[2, 4]),
    'FD32K': dict(_BASE, depth=4, channels=[32, 64, 64, 128],
                  kernel_size=3, pool_positions=[2, 4]),
    'FD64K': dict(_BASE, depth=4, channels=[32, 64, 64, 64],
                  kernel_size=3, pool_positions=[4]),
}

# XL 组: 超大 MACs 边界
GROUP_XL = {
    'XL1': dict(_BASE, depth=5, channels=[64, 64, 128, 128, 128],
                kernel_size=3, pool_positions=[2, 5]),
    'XL2': dict(_BASE, depth=5, channels=[64, 64, 128, 128, 128],
                kernel_size=5, pool_positions=[2, 5]),
}

# 均匀组: MACs 中间采样 (不同 depth/channel 组合)
GROUP_MID = {
    'MD1': dict(_BASE, depth=5, channels=[16, 32, 32, 64, 64],
                kernel_size=3, pool_positions=[2, 3, 5]),
    'MD2': dict(_BASE, depth=4, channels=[16, 32, 64, 128],
                kernel_size=3, pool_positions=[2, 4]),
    'MD3': dict(_BASE, depth=6, channels=[32, 32, 64, 64, 64, 128],
                kernel_size=3, pool_positions=[2, 4, 6]),
    'MK5': dict(_BASE, depth=3, channels=[32, 64, 128],
                kernel_size=5, pool_positions=[2, 3]),
}

EXTRA_MODELS = {}
for g in (GROUP_D, GROUP_XL, GROUP_MID):
    EXTRA_MODELS.update(g)


def main():
    print(f'{"model_id":<6}{"group":<6}{"params":>10}{"convM":>8}'
          f'{"linM":>7}{"totalM":>8}{"flatten":>8}')
    for mid, cfg in EXTRA_MODELS.items():
        cfg = validate_model_config(cfg)
        d = analyze_model_detail(build_model(cfg))
        s, c = d['summary'], d['classifier']
        grp = ('D' if mid in GROUP_D else 'XL' if mid in GROUP_XL else 'M')
        print(f'{mid:<6}{grp:<6}{s["parameters"]:>10,}'
              f'{s["conv_macs"]/1e6:>8.1f}{s["linear_macs"]/1e6:>7.2f}'
              f'{s["total_macs"]/1e6:>8.1f}{c["flatten_dimension"]:>8,}')


if __name__ == '__main__':
    main()
