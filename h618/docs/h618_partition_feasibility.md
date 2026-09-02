# H618 扩展实验与异构切分可行性报告

**日期**：2026-09-02 | **数据**：25 模型（14 legacy + 11 受控扩展），RK3566 + H618 双平台

---

## 1. 实验设计

扩展模型（11 个）按受控原则设计：

| 组 | 模型 | 设计 |
|----|------|------|
| **D: Flatten 分级** ★ | FD4K/8K/16K/32K/64K | flatten 4096→65536 严格分级，backbone [32,64,64,X] |
| XL: 大 MACs 边界 | XL1/XL2 | 537M / 1489M（k3/k5） |
| MD: 中间采样 | MD1-3, MK5 | MACs 44M~431M 均匀补充 |

## 2. 核心结果

### 2.1 双平台 M2 回归系数（T = a·Conv + b·Linear + c）

| 平台 | conv (us/M) | linear (us/M) | **linear/conv 比值** |
|------|------------|--------------|---------------------|
| RK3566 NPU | 3.0 | 975 | **325×** |
| H618 CPU | 89.5 | 1197 | **13×** |

**H1/H2 验证**：
- **H2 支持**：H618 linear 与 conv 呈稳定线性关系（13×），**无 RK3566 的 FLOAT16 GEMM 特殊执行路径惩罚**（325×）
- **H1 部分支持**：RK3566 对 linear-heavy 模型"相对低效"（浪费 NPU conv 优势）；但 RK3566 **绝对延迟仍全面领先**（conv 30×、linear 1.2× 都快）

### 2.2 FD 组双平台差距（flatten 增大）

| 模型 | flatten | RK3566 (pred) | H618 CPU | RK/H618 |
|------|--------:|--------------:|---------:|--------:|
| FD4K | 4K | 0.59ms | 14.3ms | 24.4× |
| FD32K | 32K | 1.99ms | 20.1ms | 10.1× |
| FD64K | 64K | 3.95ms | 38.7ms | 9.8× |

flatten 增大 → 双平台差距缩小（linear-heavy 模型 RK3566 相对优势变小）。

### 2.3 CPU Predictor v2（25 模型 5-fold CV）

| 模型 | Val R² | Val MAPE |
|------|--------|----------|
| M1 (MACs) | 0.925±0.067 | 8.8% |
| **M2 (Conv+Linear)** | **0.938±0.081** | **6.5%** |
| M3 (+flatten) | =M2 | =M2 |

**Q5: predictor v2 更可靠**（CV MAPE 6.5% vs v1 holdout 7.4%，且 CV 更严格）。

## 3. Partition 可行性分析（P3: Conv backbone | Flatten+Linear）

| 模型 | RK_full | RK_conv | H618_lin | hete(无传输) |
|------|--------:|--------:|---------:|-------------:|
| FD64K | 3.97ms | 1.14ms | 3.37ms | 4.52ms |
| d3_k3 | 1.54ms | 0.09ms | 1.69ms | 1.78ms |

**结论（数据支撑，如实）**：
1. RK3566 **每个算子绝对延迟都低于 H618** → 单模型延迟最小化时 **RK3566 全量最优**，T_hetero > T_RK 必然成立
2. transfer（16x16x64=16KB，USB2 ~0.4ms）只会加大差距
3. **异构切分的价值不在单模型延迟**，而在：
   - 多模型**并发/流水线**（RK3566 腾出算力跑计算密集任务）
   - RK3566 **资源受限**（大模型放不下/多路并发）
   - linear-heavy 模型卸载 H618 后 RK3566 利用率提升

## 4. 回答研究问题

**Q1**: MACs→H618 CPU 关系 25 模型仍稳定（M1 CV R²=0.925）
**Q2**: Depth/Channel/Kernel 通过 MACs 间接影响（CPU 计算量主导）
**Q3**: **是，双平台曲线不同**——RK linear 相对惩罚 325× vs H618 13×
**Q4**: 相同结构不同 composition：linear-heavy 在 RK 相对慢（FD 组证据）
**Q5**: **是**，v2 CV MAPE 6.5% < v1 7.4%
**Q6**: 无模型有绝对 cross-hardware advantage（RK 全胜）；linear-heavy 是"相对"RK 低效
**Q7**: 理论切分点 P3（conv|linear）存在，但单模型延迟无收益
**Q8**: **无理论延迟收益**（H618 任何段都慢）；收益场景 = 并发/资源分配

## 5. 决策树结论

```
Flatten/Linear penalty in RK3566?
    YES (325x vs conv)
        ↓
H618 relative advantage?
    NO (绝对延迟 RK 仍快 1.2x~30x)
        ↓
→ 不构成"单模型延迟"切分动机
→ 切分价值限于: 并发/流水线/资源受限场景 (需真实通信 benchmark 下一阶段)
```

## 6. Limitations

- FD 组 conv backbone 与 flatten 耦合（无法完全解耦 conv/linear 效应）
- RK3566 FD/XL 组为 predictor 预测（未实测，标注 predicted）
- 通信未实测（transfer 为理论估算）
- linear-heavy 占比有限（GTSRB 43 类 FC 权重上限 ~3M）

## 7. 产出

```
results/h618_baseline_summary.csv    # 14 模型双平台 baseline
results/h618_dataset.csv             # 25 模型 (14+11)
results/partition_candidates.csv     # 分段延迟估算
models/h618_cpu_latency_v2.json      # 25 模型 CV 验证 predictor
docs/figures/h618_flatten_dual_platform.png
```

## 8. 下一步建议

1. **若做异构**：定位为"RK3566 跑计算密集 + H618 跑 linear-heavy 的并发流水线"，需真实通信 benchmark
2. **通信阶段**：测 USB/网络 transfer 实测（FM 16KB~64KB 级别）
3. 或先汇合队友 accuracy 数据完成单板 Pareto 搜索
