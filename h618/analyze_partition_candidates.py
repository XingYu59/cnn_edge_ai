"""
Partition Candidate 分析 (Phase 12-19)
=======================================
对 flatten/linear-heavy 模型, 分析切分点 P3 (Conv backbone | Flatten+Linear):
  分段延迟: 用双平台 M2 系数估算 (RK3566 实测拟合 + H618 25 模型拟合)
  transfer : 切分点 FM size (C*H*W*1byte int8)

判断: T_hetero = T_RK,pre + T_transfer + T_H618,post 是否 < T_RK,full

注: 通信速率未实测, 给理论区间 (USB2.0 ~40MB/s, USB3.0 ~300MB/s 场景假设)。
"""
import os
import sys

import pandas as pd

H618_DIR = os.path.dirname(os.path.abspath(__file__))

# 双平台 M2 系数 (us/M, 来自实测拟合)
RK = {'conv': 3.0, 'linear': 975.0}        # us/M
H6 = {'conv': 89.5 * 1000 / 1e3, 'linear': 1197.0 * 1000 / 1e3}
# 修正: H618 系数单位 ms/M -> us/M
H6 = {'conv': 89.5, 'linear': 1197.0}      # us/M


def main():
    df = pd.read_csv(os.path.join(H618_DIR, 'results', 'h618_dataset.csv'))
    # 选 flatten-heavy 代表
    sel = df[df['model_id'].isin(
        ['FD16K', 'FD32K', 'FD64K', 'C1', 'V11', 'd3_k3', 'd5_k3'])]

    print('===== P3 切分: Conv backbone (RK3566) | Flatten+Linear (H618) =====')
    print(f'{"model":<7}{"convM":>7}{"linM":>6}{"RK_full":>9}{"RK_conv":>9}'
          f'{"H618_lin":>10}{"hete(无传输)":>12}{"FM@8x8":>9}{"FM@16x16":>10}')
    for _, r in sel.iterrows():
        cm, lm = r['conv_macs'] / 1e6, r['linear_macs'] / 1e6
        rk_full = RK['conv'] * cm + RK['linear'] * lm + 80   # + 固定开销us
        rk_conv = RK['conv'] * cm
        h6_lin = H6['linear'] * lm
        hete = rk_conv + h6_lin                            # us (无传输)
        fm8 = 8 * 8 * 64 * 1  # bytes (8x8x64)
        fm16 = 16 * 16 * 64 * 1
        print(f'{r["model_id"]:<7}{cm:>7.0f}{lm:>6.2f}'
              f'{rk_full/1000:>9.2f}{rk_conv/1000:>9.2f}'
              f'{h6_lin/1000:>10.3f}{hete/1000:>12.3f}'
              f'{fm8:>8}B{fm16:>9}B')

    print('\n===== transfer 时间估算 (理论) =====')
    print('  16x16x64 FM = 16KB | 32x32x64 = 64KB | 8x8x128 = 8KB')
    print('  USB2 (~40MB/s): 16KB ≈ 0.4ms | USB3 (~300MB/s): 16KB ≈ 0.05ms')

    print('\n===== 结论 (基于当前数据) =====')
    print('  1) RK3566 每个算子的绝对延迟都低于 H618 (conv 30x, linear 1.2x)')
    print('  2) T_hetero = RK_conv + H618_linear > RK_full 必然成立 '
          '(H618 linear 也慢于 RK linear)')
    print('     → 单模型延迟最小化: RK3566 全量最优, 切分无收益')
    print('  3) RK linear/conv 比值 325x vs H618 13x:')
    print('     linear-heavy 模型在 RK3566 上"相对低效" (浪费 NPU conv 优势)')
    print('  4) 切分的真实价值: 多模型并发/流水线/RK3566 资源受限时')
    print('     把 linear-heavy 模型卸载到 H618 腾出 RK3566')

    # 导出表
    rows = []
    for _, r in df.iterrows():
        cm, lm = r['conv_macs'] / 1e6, r['linear_macs'] / 1e6
        rows.append({
            'model_id': r['model_id'],
            'rk_full_ms': round((RK['conv'] * cm + RK['linear'] * lm + 80) / 1000, 3),
            'rk_conv_part_ms': round(RK['conv'] * cm / 1000, 3),
            'h618_linear_part_ms': round(H6['linear'] * lm / 1000, 3),
            'hete_no_transfer_ms': round((RK['conv'] * cm + H6['linear'] * lm) / 1000, 3),
            'note': 'hete = RK_conv + H618_linear (无传输, 理论)',
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(H618_DIR, 'results',
                            'partition_candidates.csv'), index=False)
    print(f'\n表: results/partition_candidates.csv')
    print('===== 完成 =====')


if __name__ == '__main__':
    main()
